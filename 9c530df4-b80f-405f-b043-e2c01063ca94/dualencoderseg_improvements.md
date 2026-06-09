# Improvement Plan for `DualEncoderSeg`

This document details the code modifications and enhancements to improve the segmentation performance of the **LatentSegNet v2** model.

---

## 1. Pretrained Backbone for `ImageEncoder`

Using a scratch-trained encoder on a small retina dataset leads to overfitting. We can replace it with a pretrained `ResNet34` or `EfficientNet` backbone from `torchvision.models`.

Here is the improved `ImageEncoder` utilizing a pretrained `ResNet34` backbone:

```python
import torchvision.models as models

class PretrainedImageEncoder(nn.Module):
    """
    Encodes 512×512×3 retina image → spatial latent (B, latent_ch, 16, 16).
    Uses a pretrained ResNet34 backbone to extract robust features.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        # Extract layers:
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 ch, 256×256
        self.maxpool = resnet.maxpool                                     # 64 ch, 128×128
        self.layer1 = resnet.layer1                                       # 64 ch, 128×128
        self.layer2 = resnet.layer2                                       # 128 ch, 64×64
        self.layer3 = resnet.layer3                                       # 256 ch, 32×32
        self.layer4 = resnet.layer4                                       # 512 ch, 16×16

        # Projection layer: compress 512 channels → latent_ch (128)
        self.project = nn.Sequential(
            nn.Dropout2d(cfg.dropout),
            nn.Conv2d(512, cfg.latent_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(cfg.latent_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: B×3×512×512
        x = self.stem(x)       # 256×256
        x = self.maxpool(x)    # 128×128
        x = self.layer1(x)     # 128×128
        x = self.layer2(x)     # 64×64
        x = self.layer3(x)     # 32×32
        x = self.layer4(x)     # 16×16
        return self.project(x) # B×latent_ch×16×16
```

---

## 2. Latent Alignment (VICReg) Correction

To prevent spatial distortion, VICReg's variance and covariance constraints should be calculated across the batch dimension $B$ for each spatial coordinate independently, and then averaged over the spatial grid.

```python
def corrected_vicreg_loss(
    z_pred: torch.Tensor,
    z_mask: torch.Tensor,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
) -> torch.Tensor:
    """
    VICReg loss computed across the batch dimension B (maintaining spatial structure).
    """
    B, C, H, W = z_pred.shape
    
    # 1. Invariance (MSE)
    sim = F.mse_loss(z_pred, z_mask.detach())

    # 2. Variance (hinge loss at std=1 across batch B)
    # Calculate std across batch (dim=0)
    std_p = torch.sqrt(z_pred.var(dim=0) + 1e-4) # Shape: (C, H, W)
    std_m = torch.sqrt(z_mask.var(dim=0) + 1e-4) # Shape: (C, H, W)
    
    var = F.relu(1 - std_p).mean() + F.relu(1 - std_m).mean()

    # 3. Covariance (computed over batch B, then averaged over H, W)
    def off_diag_cov_loss_spatial(z):
        # Center variables along batch dimension
        z = z - z.mean(dim=0, keepdim=True) # B×C×H×W
        
        # Reshape to easily compute batched matrix multiply: (H*W, B, C)
        z_flat = z.permute(2, 3, 0, 1).reshape(H * W, B, C)
        
        # Batch transpose and multiply: (H*W, C, B) @ (H*W, B, C) -> (H*W, C, C)
        cov = torch.bmm(z_flat.transpose(1, 2), z_flat) / (B - 1)
        
        # Penalize off-diagonal elements
        cov_sq = cov ** 2
        # Zero out the diagonal for all H*W matrices
        diag_mask = torch.eye(C, device=z.device).unsqueeze(0).expand(H * W, -1, -1)
        cov_sq = cov_sq * (1 - diag_mask)
        
        return cov_sq.sum() / (C * H * W)

    cov = off_diag_cov_loss_spatial(z_pred) + off_diag_cov_loss_spatial(z_mask)

    return sim_w * sim + var_w * var + cov_w * cov
```

---

## 3. High-Resolution Skip Connections (Residual Feature Injection)

In Stage 1 (Mask Autoencoder training), we only have access to the mask. In Stage 2, we want to inject high-resolution features from the `ImageEncoder` to recover fine-grained blood vessels. 
We can use **Residual Feature Injection**: the `MaskDecoder` accepts optional skip connections from the encoder. During Stage 1, these default to `None` (or use the `MaskEncoder`'s skips). In Stage 2, the `ImageEncoder`'s intermediate features are mapped (using $1 \times 1$ convs to match channels) and added to the decoder stages.

```python
class UpStageWithSkip(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBNReLU(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Inject skip connection using addition (residual style)
            x = x + skip
        return self.conv(x)


class ImprovedMaskDecoder(nn.Module):
    """
    Accepts optional skip connections from high-resolution layers.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.expand = ConvBNReLU(cfg.latent_ch, 256, kernel=1, padding=0)

        # 5 Upstages matching resolutions: 16 → 32 → 64 → 128 → 256 → 512
        self.up1 = UpStageWithSkip(256, 256)   # 16→32
        self.up2 = UpStageWithSkip(256, 128)   # 32→64
        self.up3 = UpStageWithSkip(128, 128)   # 64→128
        self.up4 = UpStageWithSkip(128, 64)    # 128→256
        self.up5 = UpStageWithSkip(64, 32)     # 256→512
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: torch.Tensor, skips: Optional[list] = None) -> torch.Tensor:
        # skips is a list of features at resolutions [32, 64, 128, 256, 512]
        x = self.expand(z)
        x = self.up1(x, skips[0] if skips is not None else None)
        x = self.up2(x, skips[1] if skips is not None else None)
        x = self.up3(x, skips[2] if skips is not None else None)
        x = self.up4(x, skips[3] if skips is not None else None)
        x = self.up5(x, skips[4] if skips is not None else None)
        return self.out(x)
```

---

## 4. Bilinear Downsampling for Auxiliary target

Change the auxiliary target downsampling to use bilinear interpolation to prevent losing thin vessels:

```python
# Inside SegmentationLoss.forward:
# Downsample target mask using bilinear interpolation to get soft targets
aux_target = F.interpolate(target_mask, size=(32, 32), mode='bilinear', align_corners=False)
```

And update the auxiliary head's loss from Tversky loss (designed for binary targets) to a soft BCE/MSE loss, or binarize the soft targets using a threshold.

---

## 5. Summary of Recommended Pipeline Changes

1. **Load Pretrained ResNet34** inside the `ImageEncoder`.
2. **Apply Spatial-preserving VICReg** (the batched spatial covariance version above).
3. **Connect intermediate layers** from the ResNet encoder (stem, layer1, layer2, layer3, layer4) via $1 \times 1$ convs to the corresponding `MaskDecoder` upstages.
4. **Use bilinear interpolation** for the auxiliary head target.


# ===================="""
LatentSegNet v2 — Improved Latent-Space Bridging Segmentation Model
=====================================================================

Changes from v1:
  1. Spatial Latent (16×16 grid instead of flat 2048-vector)
     → Preserves spatial structure through the bottleneck
     → Thin vessels retain positional information
  2. VICReg-style Latent Alignment Loss
     → Replaces naive MSE; prevents dimensional collapse
     → Forces decorrelated, variance-stabilized latent dims
  3. Tversky + Boundary Loss
     → Tversky: tunable FN/FP asymmetry for vessel imbalance
     → Boundary: 5× weight on vessel edges (1-2px vessels)
  4. Auxiliary Segmentation Head
     → 32×32 prediction head directly after MappingMLP
     → Short gradient path prevents vanishing deep in decoder
  5. LR Warmup + Cosine Decay
     → Stable Stage 2 cold-start

Architecture:
  Image/Mask  →  5 DownStages  →  AdaptiveAvgPool(16×16)  →  spatial latent (B,C,16,16)
  MappingMLP  →  ConvMLP on spatial grid
  Decoder     →  7 UpStages from (B,C,16,16)  →  512×512×1
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass
class Config:
    # Spatial latent dimensions — 16×16×128 = 32768 structured values
    # vs old flat 4×4×256→2048 (information preserved, spatially organized)
    spatial_h: int = 16
    spatial_w: int = 16
    latent_ch: int = 128          # channels in spatial latent grid

    image_size: int = 512
    dropout: float = 0.1

    # Loss weights
    lambda_vicreg_sim: float = 25.0
    lambda_vicreg_var: float = 25.0
    lambda_vicreg_cov: float = 1.0
    lambda_tversky: float = 1.5
    lambda_boundary: float = 1.0
    lambda_aux: float = 0.3       # auxiliary 32×32 head weight

    # Tversky: alpha=FP penalty, beta=FN penalty
    # beta > alpha → missing vessels hurt more than false positives
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7

    # Training
    stage1_epochs: int = 50
    stage2_epochs: int = 100
    stage1_lr: float = 1e-3
    stage2_lr: float = 3e-4
    warmup_epochs: int = 10       # Stage 2 LR warmup
    latent_warmup_epochs: int = 15


cfg = Config()
print("Config loaded.")
print(f"  Spatial latent: {cfg.spatial_h}×{cfg.spatial_w}×{cfg.latent_ch}"
      f"  ({cfg.spatial_h*cfg.spatial_w*cfg.latent_ch:,} values)")
print(f"  Old flat latent was: 2048 values (no spatial structure)")


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class ResBlock(nn.Module):
    """SE-ResBlock — same as v1, unchanged."""
    def __init__(self, ch, se_ratio=4):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(ch, ch),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        hidden = max(ch // se_ratio, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, hidden, bias=False), nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=False), nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv(x)
        scale = self.se(res).view(res.size(0), -1, 1, 1)
        return self.relu(x + res * scale)


class DownStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(ConvBNReLU(in_ch, out_ch, stride=2), ResBlock(out_ch))
    def forward(self, x): return self.block(x)


class UpStage(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBNReLU(in_ch, out_ch)
    def forward(self, x): return self.conv(self.up(x))


print("Building blocks defined.")


class MaskEncoder(nn.Module):
    """
    Encodes 512×512×1 mask → spatial latent (B, latent_ch, 16, 16).

    KEY CHANGE from v1:
      Old: AdaptiveAvgPool(4,4) → Flatten → Linear(4096, 2048)
           → 1D vector, ALL spatial info destroyed
      New: AdaptiveAvgPool(16,16) → Conv1×1(256, latent_ch)
           → 2D spatial grid, position of vessels PRESERVED

    Why 16×16?
      After 5 stride-2 downsamples: 512→256→128→64→32→16
      Pool to 16×16 means no spatial compression at the bottleneck —
      just channel compression (256 → 128). Information-theoretically sound.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.stem  = ConvBNReLU(1, 32, stride=1)   # 512
        self.down1 = DownStage(32, 64)              # 256
        self.down2 = DownStage(64, 128)             # 128
        self.down3 = DownStage(128, 256)            # 64
        self.down4 = DownStage(256, 256)            # 32
        self.down5 = DownStage(256, 256)            # 16

        # Spatial pool: keep 16×16 grid, compress channels
        self.pool    = nn.AdaptiveAvgPool2d((cfg.spatial_h, cfg.spatial_w))
        self.project = nn.Sequential(
            nn.Conv2d(256, cfg.latent_ch, 1, bias=False),
            nn.BatchNorm2d(cfg.latent_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """B×1×512×512  →  B×latent_ch×16×16"""
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return self.project(self.pool(x))


class MaskDecoder(nn.Module):
    """
    Reconstructs 512×512×1 from spatial latent (B, latent_ch, 16, 16).

    KEY CHANGE from v1:
      Old: Linear(2048, 4*4*256) → reshape(4,4,256) → 7 UpStages
           → decoder had to learn spatial positions from scratch
      New: latent already is (B, 128, 16, 16), expand channels → 7 UpStages
           → decoder decodes, not reconstructs spatial positions

    Path: 16→32→64→128→256→512 (5 up-stages sufficient, 2 more for quality)
    """
    def __init__(self, cfg: Config):
        super().__init__()
        # Expand latent channels back up
        self.expand = ConvBNReLU(cfg.latent_ch, 256, kernel=1, padding=0)

        self.up1 = UpStage(256, 256)   # 16→32
        self.up2 = UpStage(256, 128)   # 32→64
        self.up3 = UpStage(128, 128)   # 64→128
        self.up4 = UpStage(128, 64)    # 128→256
        self.up5 = UpStage(64, 32)     # 256→512
        self.out  = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """B×latent_ch×16×16  →  B×1×512×512 (logits)"""
        x = self.expand(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.out(x)


class MaskAutoencoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = MaskEncoder(cfg)
        self.decoder = MaskDecoder(cfg)

    def forward(self, mask: torch.Tensor):
        z = self.encoder(mask)
        return self.decoder(z), z

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("MaskEncoder frozen.")


print("MaskAutoencoder defined.")


class ImageEncoder(nn.Module):
    """
    Encodes 512×512×3 retina image → spatial latent (B, latent_ch, 16, 16).

    Mirrors MaskEncoder architecture for symmetric mapping.
    Larger initial channels (3→64 instead of 1→32) because RGB images
    contain more information than binary masks.

    Drop-in replacement note:
      To use a pretrained backbone (EfficientNet, ResNet), replace
      down1-down5 with backbone feature extraction and add a
      Conv2d head to project to (latent_ch, 16, 16).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.stem  = ConvBNReLU(3, 64, stride=1)   # 512
        self.down1 = DownStage(64, 128)             # 256
        self.down2 = DownStage(128, 256)            # 128
        self.down3 = DownStage(256, 512)            # 64
        self.down4 = DownStage(512, 512)            # 32
        self.down5 = DownStage(512, 256)            # 16  ← channel bottleneck

        self.pool    = nn.AdaptiveAvgPool2d((cfg.spatial_h, cfg.spatial_w))
        self.project = nn.Sequential(
            nn.Dropout2d(cfg.dropout),
            nn.Conv2d(256, cfg.latent_ch, 1, bias=False),
            nn.BatchNorm2d(cfg.latent_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """B×3×512×512  →  B×latent_ch×16×16"""
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        return self.project(self.pool(x))


print("ImageEncoder defined.")


class SpatialMappingNet(nn.Module):
    """
    Maps image spatial latent → mask spatial latent.
    Both are (B, latent_ch, 16, 16) tensors.

    KEY CHANGE from v1:
      Old: Linear(2048→4096→4096→2048) — purely channel-wise MLP,
           spatially blind (each of 16×16 positions processed identically)
      New: 1×1 convolutions = per-position MLPs + 3×3 convs for spatial mixing
           → each position can look at its 8 neighbors when translating

    Architecture: expand → mix spatially → contract → residual
    The 3×3 conv is the critical addition — it lets vessel continuity
    (a vessel at position (i,j) is likely to have a vessel at (i,j±1))
    be learned explicitly.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.latent_ch
        hidden = ch * 2   # 256 channels in hidden space

        self.net = nn.Sequential(
            # 1×1: channel expansion (position-wise)
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),

            # 3×3: spatial mixing (cross-position context)
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),  # depthwise
            nn.Conv2d(hidden, hidden, 1, bias=False),                             # pointwise
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(cfg.dropout),

            # 1×1: project back
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        # Residual path (stable gradients from day 1)
        self.residual = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """B×C×16×16  →  B×C×16×16"""
        return self.net(z) + self.residual(z)


class AuxHead(nn.Module):
    """
    Auxiliary head: predicts a 32×32 mask directly from the mapped latent.

    PURPOSE: Creates a SHORT gradient path back to ImageEncoder + MappingNet.
    Without this, the gradient must flow: loss → decoder (7 UpStages) →
    MappingNet → ImageEncoder. With aux loss, there's a direct path.

    At inference: this head is unused (overhead = 0).
    During training: contributes 0.3× of total loss.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.latent_ch
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16→32
            nn.Conv2d(ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """B×C×16×16  →  B×1×32×32 (logits)"""
        return self.head(z)


print("SpatialMappingNet + AuxHead defined.")


class SpatialMappingNet(nn.Module):
    """
    Maps image spatial latent → mask spatial latent.
    Both are (B, latent_ch, 16, 16) tensors.

    KEY CHANGE from v1:
      Old: Linear(2048→4096→4096→2048) — purely channel-wise MLP,
           spatially blind (each of 16×16 positions processed identically)
      New: 1×1 convolutions = per-position MLPs + 3×3 convs for spatial mixing
           → each position can look at its 8 neighbors when translating

    Architecture: expand → mix spatially → contract → residual
    The 3×3 conv is the critical addition — it lets vessel continuity
    (a vessel at position (i,j) is likely to have a vessel at (i,j±1))
    be learned explicitly.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.latent_ch
        hidden = ch * 2   # 256 channels in hidden space

        self.net = nn.Sequential(
            # 1×1: channel expansion (position-wise)
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),

            # 3×3: spatial mixing (cross-position context)
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),  # depthwise
            nn.Conv2d(hidden, hidden, 1, bias=False),                             # pointwise
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Dropout2d(cfg.dropout),

            # 1×1: project back
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        # Residual path (stable gradients from day 1)
        self.residual = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """B×C×16×16  →  B×C×16×16"""
        return self.net(z) + self.residual(z)


class AuxHead(nn.Module):
    """
    Auxiliary head: predicts a 32×32 mask directly from the mapped latent.

    PURPOSE: Creates a SHORT gradient path back to ImageEncoder + MappingNet.
    Without this, the gradient must flow: loss → decoder (7 UpStages) →
    MappingNet → ImageEncoder. With aux loss, there's a direct path.

    At inference: this head is unused (overhead = 0).
    During training: contributes 0.3× of total loss.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.latent_ch
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16→32
            nn.Conv2d(ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """B×C×16×16  →  B×1×32×32 (logits)"""
        return self.head(z)


print("SpatialMappingNet + AuxHead defined.")


class LatentSegNet(nn.Module):
    """
    Full segmentation model (Stage 2).

    Forward pass (training):
        image → ImageEncoder → SpatialMappingNet → MaskDecoder → pred_logits
                                               ↓
                                           AuxHead → aux_32x32_logits
        mask  → MaskEncoder (frozen) → z_mask (for VICReg alignment)

    Forward pass (inference):
        image → ImageEncoder → SpatialMappingNet → MaskDecoder → logits
        (AuxHead and MaskEncoder not called)
    """
    def __init__(self, cfg: Config, pretrained_mae: MaskAutoencoder = None):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg)
        self.mapping       = SpatialMappingNet(cfg)
        self.aux_head      = AuxHead(cfg)

        if pretrained_mae is not None:
            self.mask_decoder = pretrained_mae.decoder
            self.mask_encoder = pretrained_mae.encoder
            pretrained_mae.freeze_encoder()
        else:
            # Cold-start fallback
            self.mask_encoder = MaskEncoder(cfg)
            self.mask_decoder = MaskDecoder(cfg)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """image → predicted mask latent grid (B, C, 16, 16)."""
        return self.mapping(self.image_encoder(image))

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None):
        z_pred = self.encode_image(image)
        pred_logits = self.mask_decoder(z_pred)

        if mask is not None:
            # Auxiliary prediction (short gradient path)
            aux_logits = self.aux_head(z_pred)

            # GT latent (frozen encoder — no_grad guaranteed by requires_grad=False)
            z_mask = self.mask_encoder(mask)
            return pred_logits, aux_logits, z_pred, z_mask

        return pred_logits


print("LatentSegNet v2 defined.")

# ── Pixel Losses ────────────────────────────────────────────────

def tversky_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,     # FP penalty
    beta: float = 0.7,      # FN penalty (missing vessels hurt more)
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Tversky loss — asymmetric Dice.
    beta > alpha: we MUCH more care about not missing vessels than
    creating false positives. Flip for disc segmentation.

    When alpha=beta=0.5, reduces exactly to Dice loss.
    """
    pred = torch.sigmoid(pred_logits)
    tp = (pred * target).sum(dim=(2, 3))
    fp = (pred * (1 - target)).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky).mean()


def boundary_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Boundary-weighted BCE.
    Detects vessel edges in GT mask via Sobel, then applies 5× weight
    to the BCE loss at boundary pixels.

    Critical for 1-2px thin vessels: standard losses treat a 1px miss
    on a thick vessel the same as a 1px miss on a thin vessel.
    Boundary loss makes thin vessel errors proportionally more costly.
    """
    B = target.size(0)
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=target.device
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    edge = (
        F.conv2d(target, sobel_x, padding=1).abs() +
        F.conv2d(target, sobel_y, padding=1).abs()
    ).clamp(0, 1)  # B×1×512×512, 1 at boundaries

    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    weighted = bce * (1 + 4 * edge)   # 5× at boundaries, 1× elsewhere
    return weighted.mean()


# ── Latent Alignment Loss (VICReg) ──────────────────────────────

def vicreg_loss(
    z_pred: torch.Tensor,
    z_mask: torch.Tensor,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
) -> torch.Tensor:
    """
    VICReg (Variance-Invariance-Covariance Regularization).

    Three terms working together:
      1. Invariance (sim): MSE between z_pred and z_mask.
         → Pulls predicted latent toward GT latent. (Same as v1 MSE loss)

      2. Variance (var): Each latent dim should have std ≈ 1 across batch.
         → Prevents mode collapse (all masks mapping to same latent).
         → v1 had NO protection against this failure mode.

      3. Covariance (cov): Off-diagonal cov matrix entries → 0.
         → Forces each dim to encode different information.
         → v1 could waste dims on redundant information.

    Note: z tensors are (B, C, 16, 16) → we flatten spatial before VICReg.
    Each (b, h, w) triple is one "sample" in the batch dimension.
    """
    # Flatten spatial: (B, C, 16, 16) → (B*16*16, C)
    B, C, H, W = z_pred.shape
    zp = z_pred.permute(0, 2, 3, 1).reshape(-1, C)
    zm = z_mask.permute(0, 2, 3, 1).reshape(-1, C)
    N  = zp.shape[0]

    # 1. Invariance
    sim = F.mse_loss(zp, zm.detach())

    # 2. Variance — hinge at std=1 (no reward for being >1)
    std_p = torch.sqrt(zp.var(dim=0) + 1e-4)
    std_m = torch.sqrt(zm.var(dim=0) + 1e-4)
    var = F.relu(1 - std_p).mean() + F.relu(1 - std_m).mean()

    # 3. Covariance — penalize off-diagonal entries
    def off_diag_cov_loss(z):
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)
        # Zero out diagonal (we want those to be large)
        off = cov ** 2
        off.fill_diagonal_(0)
        return off.sum() / C

    cov = off_diag_cov_loss(zp) + off_diag_cov_loss(zm)

    return sim_w * sim + var_w * var + cov_w * cov


# ── Combined Loss ────────────────────────────────────────────────

class SegmentationLoss(nn.Module):
    """
    Four-term combined loss for Stage 2.

    Term            Weight  Purpose
    ──────────────────────────────────────────────────────────────
    VICReg latent   λ_v     Align + stabilize latent space
    Tversky pixel   λ_t     Vessel recall bias (FN >> FP)
    Boundary pixel  λ_b     Thin vessel edge precision
    Aux 32×32       λ_a     Short gradient path regularizer
    """
    def __init__(self, cfg: Config, lambda_vicreg: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.lv = lambda_vicreg  # curriculum-controlled from outside

    def forward(self, pred_logits, aux_logits, z_pred, z_mask, target_mask):
        # VICReg on spatial latent
        l_vic = vicreg_loss(
            z_pred, z_mask,
            sim_w=self.cfg.lambda_vicreg_sim,
            var_w=self.cfg.lambda_vicreg_var,
            cov_w=self.cfg.lambda_vicreg_cov,
        )

        # Pixel losses
        l_tvk = tversky_loss(
            pred_logits, target_mask,
            alpha=self.cfg.tversky_alpha,
            beta=self.cfg.tversky_beta,
        )
        l_bnd = boundary_loss(pred_logits, target_mask)

        # Auxiliary head (32×32 downsampled GT)
        aux_target = F.interpolate(target_mask, size=(32, 32), mode='nearest')
        l_aux = tversky_loss(aux_logits, aux_target,
                              alpha=self.cfg.tversky_alpha,
                              beta=self.cfg.tversky_beta)

        total = (
            self.lv * l_vic +
            self.cfg.lambda_tversky * l_tvk +
            self.cfg.lambda_boundary * l_bnd +
            self.cfg.lambda_aux * l_aux
        )

        breakdown = {
            'vicreg': l_vic.item(),
            'tversky': l_tvk.item(),
            'boundary': l_bnd.item(),
            'aux': l_aux.item(),
        }
        return total, breakdown


class AutoencoderLoss(nn.Module):
    """Stage 1: Tversky + Boundary (no latent term needed)."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, recon_logits, target_mask):
        return (
            tversky_loss(recon_logits, target_mask,
                          alpha=self.cfg.tversky_alpha, beta=self.cfg.tversky_beta)
            + boundary_loss(recon_logits, target_mask)
        )


print("All loss functions defined.")

@torch.no_grad()
def compute_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """IoU, Dice, pixel accuracy, sensitivity, specificity."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2, 3))
    s  = 1e-5
    return {
        'iou':         ((tp+s) / (tp+fp+fn+s)).mean().item(),
        'dice':        ((2*tp+s) / (2*tp+fp+fn+s)).mean().item(),
        'acc':         (pred == target).float().mean().item(),
        'sensitivity': ((tp+s) / (tp+fn+s)).mean().item(),  # recall
        'specificity': ((tn+s) / (tn+fp+s)).mean().item(),
    }


print("Metrics defined.")

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Linear warmup → cosine decay.
    Warmup: prevents Stage 2 cold-start divergence (image encoder
    starts with random weights hitting a pretrained decoder).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress)) # cosine decay
    return LambdaLR(optimizer, lr_lambda)


print("LR scheduler defined.")
def pretrain_mask_autoencoder(
    mae: MaskAutoencoder,
    train_loader,
    val_loader=None,
    cfg: Config = None,
    device: str = 'cuda',
):
    """
    Stage 1: Learn the mask manifold.
    Target: val Dice > 0.90 before proceeding to Stage 2.
    Uses Tversky + Boundary loss (no latent term in Stage 1).

    Stopping criterion: if best_val_dice < 0.85 after all epochs,
    print a warning and suggest increasing latent_ch.
    """
    if cfg is None: cfg = Config()
    mae = mae.to(device)
    criterion = AutoencoderLoss(cfg)
    optimizer = AdamW(mae.parameters(), lr=cfg.stage1_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs)

    print("=" * 60)
    print("STAGE 1 — Pretraining Mask Autoencoder")
    print(f"  Spatial latent: {cfg.spatial_h}×{cfg.spatial_w}×{cfg.latent_ch}")
    print("=" * 60)

    best_dice = 0.0
    for epoch in range(1, cfg.stage1_epochs + 1):
        mae.train()
        total_loss = 0.0
        for batch in train_loader:
            masks = batch[1] if isinstance(batch, (list, tuple)) else batch
            masks = masks.float().to(device)
            optimizer.zero_grad()
            recon, _ = mae(masks)
            loss = criterion(recon, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == cfg.stage1_epochs:
            avg = total_loss / len(train_loader)
            val_str = ""
            if val_loader is not None:
                mae.eval()
                mlist = []
                with torch.no_grad():
                    for batch in val_loader:
                        masks = batch[1] if isinstance(batch, (list, tuple)) else batch
                        masks = masks.float().to(device)
                        recon, _ = mae(masks)
                        mlist.append(compute_metrics(recon, masks))
                d = sum(m['dice'] for m in mlist) / len(mlist)
                best_dice = max(best_dice, d)
                val_str = f"  val Dice={d:.4f}"
            print(f"Epoch {epoch:3d}/{cfg.stage1_epochs}  loss={avg:.4f}{val_str}")

    if best_dice < 0.85 and val_loader is not None:
        print(f"\n⚠  WARNING: best val Dice={best_dice:.4f} < 0.85.")
        print("   → Increase cfg.latent_ch (try 256) or add skip connections.")
    print("Stage 1 complete.\n")
    return mae
def train_segmentation(
    model: LatentSegNet,
    train_loader,
    val_loader=None,
    cfg: Config = None,
    device: str = 'cuda',
):
    """
    Stage 2: Train ImageEncoder + SpatialMappingNet + MaskDecoder.
    MaskEncoder is frozen (no gradient, no BN stat update).

    VICReg curriculum:
      Epochs 1..latent_warmup:  lambda_vicreg=2.0  (alignment dominant)
      Epochs warmup+1..:        lambda_vicreg=0.5  (pixel losses take over)
    """
    if cfg is None: cfg = Config()
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    print(f"Frozen parameters:    "
          f"{sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    optimizer = AdamW(trainable, lr=cfg.stage2_lr, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.warmup_epochs, cfg.stage2_epochs)

    print("\n" + "=" * 60)
    print("STAGE 2 — Training Segmentation Model")
    print("=" * 60)

    for epoch in range(1, cfg.stage2_epochs + 1):
        lv = 2.0 if epoch <= cfg.latent_warmup_epochs else 0.5
        criterion = SegmentationLoss(cfg, lambda_vicreg=lv)

        model.train()
        model.mask_encoder.eval()   # frozen BN stats
        total_loss = 0.0
        breakdown_acc = {'vicreg': 0., 'tversky': 0., 'boundary': 0., 'aux': 0.}

        for images, masks in train_loader:
            images = images.float().to(device)
            masks  = masks.float().to(device)

            optimizer.zero_grad()
            pred_logits, aux_logits, z_pred, z_mask = model(images, masks)
            loss, bd = criterion(pred_logits, aux_logits, z_pred, z_mask, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            for k in breakdown_acc: breakdown_acc[k] += bd[k]

        scheduler.step()

        if epoch % 10 == 0 or epoch == cfg.stage2_epochs:
            n = len(train_loader)
            avg = total_loss / n
            bd_str = "  " + "  ".join(f"{k}={v/n:.3f}" for k,v in breakdown_acc.items())

            val_str = ""
            if val_loader is not None:
                model.eval()
                mlist = []
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.float().to(device)
                        masks  = masks.float().to(device)
                        pred = model(images)
                        mlist.append(compute_metrics(pred, masks))
                dice = sum(m['dice'] for m in mlist) / len(mlist)
                iou  = sum(m['iou']  for m in mlist) / len(mlist)
                sens = sum(m['sensitivity'] for m in mlist) / len(mlist)
                val_str = f"  | val Dice={dice:.4f}  IoU={iou:.4f}  Sens={sens:.4f}"

            print(f"Ep {epoch:3d}/{cfg.stage2_epochs}  loss={avg:.4f}  lv={lv:.1f}"
                  f"{bd_str}{val_str}")

    print("Stage 2 complete.\n")
    return model

@torch.no_grad()
def predict(
    model: LatentSegNet,
    image: torch.Tensor,
    threshold: float = 0.5,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Produce binary segmentation mask.
    AuxHead is NOT called at inference — zero overhead.

    Returns: B×512×512 uint8 tensor (or 512×512 if single image)
    """
    model.eval()
    squeeze = image.dim() == 3
    if squeeze: image = image.unsqueeze(0)
    image = image.to(device)
    logits = model(image)                        # B×1×512×512
    binary = (torch.sigmoid(logits) > threshold).squeeze(1).cpu().to(torch.uint8)
    return binary.squeeze(0) if squeeze else binary


@torch.no_grad()
def predict_prob(
    model: LatentSegNet,
    image: torch.Tensor,
    device: str = 'cuda',
) -> torch.Tensor:
    """Returns probability map (float32) — useful for threshold tuning."""
    model.eval()
    squeeze = image.dim() == 3
    if squeeze: image = image.unsqueeze(0)
    prob = torch.sigmoid(model(image.to(device))).squeeze(1).cpu()
    return prob.squeeze(0) if squeeze else prob


print("Inference helpers defined.")
from pathlib import Path

# Define the base directory
base_path = Path("/kaggle/input/datasets/abdallahwagih/retina-blood-vessel/Data/train")

# Use glob to find all .png files recursively or in specific folders
image_paths = sorted(list((base_path / "image").glob("*.png")))
mask_paths = sorted(list((base_path / "mask").glob("*.png")))

# Print the first 5 paths to verify
print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

"""
--- Data Shapes ---
Image shape: (512, 512, 3)
Mask shape:  (512, 512)
"""

# Define the specific paths provided
img_path = "/kaggle/input/datasets/abdallahwagih/retina-blood-vessel/Data/train/image/0.png"
mask_path = "/kaggle/input/datasets/abdallahwagih/retina-blood-vessel/Data/train/mask/0.png"

# Load the images to get shapes
img = Image.open(img_path)
mask = Image.open(mask_path)

img_array = np.array(img)
mask_array = np.array(mask)

# 1. Print Data Shapes
print(f"--- Data Shapes ---")
print(f"Image shape: {img_array.shape}")
print(f"Mask shape:  {mask_array.shape}")

# 2. List Paths
print(f"\n--- File Paths ---")
print(f"Image path: {img_path}")
print(f"Mask path:  {mask_path}")

# 3. Visualize Image and Mask
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title(f"Original Image\n{os.path.basename(img_path)}")
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title(f"Ground Truth Mask\n{os.path.basename(mask_path)}")
axes[1].axis('off')

plt.tight_layout()
plt.show()
class RetinaDataset(torch.utils.data.Dataset):
    """
    Retina segmentation dataset (DRIVE / STARE / CHASE_DB1).

    Expected structure:
        data/
          images/  *.png  (512×512 RGB)
          masks/   *.png  (512×512 binary, 0 or 255)

    Augmentations applied IDENTICALLY to image+mask:
      - Random horizontal/vertical flip
      - Random rotation ±30°
    Applied to IMAGE ONLY:
      - ColorJitter (brightness, contrast, saturation)
      - Gaussian blur
    """
    def __init__(self, image_paths, mask_paths, augment: bool = False):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.augment     = augment

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as TF
        import random

        img  = Image.open(self.image_paths[idx]).convert('RGB').resize((512,512))
        mask = Image.open(self.mask_paths[idx]).convert('L').resize((512,512))

        if self.augment:
            # Shared spatial transforms
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-30, 30)
            img  = TF.rotate(img,  angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

            # Image-only photometric
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img,   random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

        img  = TF.to_tensor(img)           # 3×512×512, [0,1]
        mask = TF.to_tensor(mask)          # 1×512×512, [0,1]
        mask = (mask > 0.5).float()        # binarize
        return img, mask


print("RetinaDataset skeleton defined.")
def count_parameters(model: nn.Module) -> None:
    rows = []
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append((name, total, trainable))

    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'Module':<25} {'Total':>12} {'Trainable':>12}")
    print("-" * 52)
    for name, tot, tr in rows:
        frozen_str = " [frozen]" if tr == 0 else ""
        print(f"{name:<25} {tot:>12,} {tr:>12,}{frozen_str}")
    print("-" * 52)
    print(f"{'TOTAL':<25} {total_all:>12,} {trainable_all:>12,}")
    print()
def sanity_check(cfg: Config = None, device: str = 'cpu'):
    """
    Full forward-pass shape validation for both stages + inference.
    Run this before any training — catches shape bugs instantly.
    """
    if cfg is None: cfg = Config()
    print(f"Sanity check on {device}  |  spatial latent: "
          f"{cfg.spatial_h}×{cfg.spatial_w}×{cfg.latent_ch}\n")

    B   = 2
    img  = torch.randn(B, 3, 512, 512).to(device)
    mask = torch.randint(0, 2, (B, 1, 512, 512)).float().to(device)

    # ── Stage 1 ──────────────────────────────────────────────
    mae = MaskAutoencoder(cfg).to(device)
    recon, z = mae(mask)
    assert recon.shape == (B, 1, 512, 512),               f"recon: {recon.shape}"
    assert z.shape     == (B, cfg.latent_ch,
                            cfg.spatial_h, cfg.spatial_w), f"z: {z.shape}"
    print(f"[Stage 1]  recon={tuple(recon.shape)}  z={tuple(z.shape)}  ✓")

    # ── Stage 2 ──────────────────────────────────────────────
    mae.freeze_encoder()
    model = LatentSegNet(cfg, pretrained_mae=mae).to(device)

    try:
        from torchinfo import summary
        summary(model, input_size=(1, 3, 512, 512), device=device)
    except ImportError:
        print("torchinfo not installed. Printing basic structure:")
        print(model)
    pred, aux, z_pred, z_mask = model(img, mask)

    assert pred.shape   == (B, 1, 512, 512),               f"pred: {pred.shape}"
    assert aux.shape    == (B, 1, 32, 32),                 f"aux: {aux.shape}"
    assert z_pred.shape == (B, cfg.latent_ch,
                             cfg.spatial_h, cfg.spatial_w), f"z_pred: {z_pred.shape}"
    print(f"[Stage 2]  pred={tuple(pred.shape)}  aux={tuple(aux.shape)}"
          f"  z_pred={tuple(z_pred.shape)}  ✓")

    # ── Loss ─────────────────────────────────────────────────
    criterion = SegmentationLoss(cfg, lambda_vicreg=2.0)
    loss, bd  = criterion(pred, aux, z_pred, z_mask, mask)
    assert loss.item() > 0, "Loss should be positive"
    print(f"[Loss]     total={loss.item():.4f}  breakdown={bd}  ✓")

    # ── Inference ────────────────────────────────────────────
    binary = predict(model, img, device=device)
    assert binary.shape == (B, 512, 512),                  f"binary: {binary.shape}"
    print(f"[Infer]    binary={tuple(binary.shape)}  ✓")

    count_parameters(model)
    print("All checks passed.\n")
def full_pipeline(
    train_loader,
    val_loader=None,
    cfg: Config = None,
    device: str = 'cuda',
    checkpoint_path: str = 'latentsegnet_v2.pt',
):
    """
    Orchestrates Stage 1 → Stage 2 → save checkpoint.

    Usage:
        cfg = Config()
        train_ds = RetinaDataset(img_paths_train, mask_paths_train, augment=True)
        val_ds   = RetinaDataset(img_paths_val,   mask_paths_val,   augment=False)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=2)
        model = full_pipeline(train_loader, val_loader, cfg, device='cuda')
    """
    if cfg is None: cfg = Config()

    # Stage 1
    mae = MaskAutoencoder(cfg)
    pretrain_mask_autoencoder(mae, train_loader, val_loader, cfg, device)

    # Stage 2
    model = LatentSegNet(cfg, pretrained_mae=mae)
    train_segmentation(model, train_loader, val_loader, cfg, device)

    # Save
    torch.save({
        'model_state': model.state_dict(),
        'cfg': cfg,
    }, checkpoint_path)
    print(f"Checkpoint saved → {checkpoint_path}")

    return model


def load_model(checkpoint_path: str, device: str = 'cpu') -> LatentSegNet:
    """Load a saved model for inference."""
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt['cfg']
    model = LatentSegNet(cfg)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model
if __name__ == '__main__':
    cfg = Config()
    sanity_check(cfg, device='cpu')

    # %% [code]
    

    print("\nSummary of improvements over v1:")
    # improvements = [
    #     ("Spatial Latent 16×16",
    #      "Preserves position of vessels. Old 1D vector destroyed all spatial info."),
    #     ("SpatialMappingNet (Conv-based)",
    #      "3×3 depthwise conv enables cross-position context (vessel continuity)."),
    #     ("VICReg Alignment Loss",
    #      "3-term loss prevents latent collapse + forces decorrelated dims."),
    #     ("Tversky Loss (α=0.3, β=0.7)",
    #      "Asymmetric: penalizes missed vessels 2.3× more than false positives."),
    #     ("Boundary-Weighted Loss",
    #      "5× weight at vessel edges catches 1-2px thin vessel misses."),
    #     ("Auxiliary 32×32 Head",
    #      "Short gradient path; removed at inference with zero overhead."),
    #     ("LR Warmup (10 epochs)",
    #      "Prevents Stage 2 divergence from random image encoder init."),
    #     ("Sensitivity metric",
    #      "Added vessel recall to validation metrics — more informative than Dice alone."),
    # ]
    # for title, desc in improvements:
    #     print(f"\n  ✦ {title}")
    #     print(f"    {desc}")
 [code] {"execution":{"iopub.status.busy":"2026-06-07T17:23:08.157283Z","iopub.execute_input":"2026-06-07T17:23:08.157594Z","iopub.status.idle":"2026-06-07T17:23:08.168901Z","shell.execute_reply.started":"2026-06-07T17:23:08.157567Z","shell.execute_reply":"2026-06-07T17:23:08.168234Z"}}
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Use the paths you already defined in CELL 13
base_path = Path("/kaggle/input/datasets/abdallahwagih/retina-blood-vessel/Data/train")
image_paths = sorted(list((base_path / "image").glob("*.png")))
mask_paths  = sorted(list((base_path / "mask").glob("*.png")))

assert len(image_paths) == len(mask_paths)
print(f"Total samples: {len(image_paths)}")

# Split 80/20
split = int(0.8 * len(image_paths))
train_images, val_images = image_paths[:split], image_paths[split:]
train_masks,  val_masks  = mask_paths[:split],  mask_paths[split:]

train_ds = RetinaDataset(train_images, train_masks, augment=True)
val_ds   = RetinaDataset(val_images,   val_masks,   augment=False)

batch_size = 4   # adjust based on GPU memory
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
 [code] {"execution":{"iopub.status.busy":"2026-06-07T17:23:08.169948Z","iopub.execute_input":"2026-06-07T17:23:08.170702Z","iopub.status.idle":"2026-06-07T17:23:08.189095Z","shell.execute_reply.started":"2026-06-07T17:23:08.170676Z","shell.execute_reply":"2026-06-07T17:23:08.188240Z"}}
def train_stage1_with_history(mae, train_loader, val_loader, cfg, device):
    """Returns history dict: train_loss, val_dice, val_iou, val_sens"""
    mae = mae.to(device)
    criterion = AutoencoderLoss(cfg)
    optimizer = AdamW(mae.parameters(), lr=cfg.stage1_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs)

    history = {'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_sens': []}

    for epoch in range(1, cfg.stage1_epochs + 1):
        mae.train()
        total_loss = 0.0
        for batch in train_loader:
            masks = batch[1] if isinstance(batch, (list, tuple)) else batch
            masks = masks.float().to(device)
            optimizer.zero_grad()
            recon, _ = mae(masks)
            loss = criterion(recon, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        # Validation
        if val_loader is not None:
            mae.eval()
            dice_list, iou_list, sens_list = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    masks = batch[1] if isinstance(batch, (list, tuple)) else batch
                    masks = masks.float().to(device)
                    recon, _ = mae(masks)
                    metrics = compute_metrics(recon, masks)
                    dice_list.append(metrics['dice'])
                    iou_list.append(metrics['iou'])
                    sens_list.append(metrics['sensitivity'])
            history['val_dice'].append(np.mean(dice_list))
            history['val_iou'].append(np.mean(iou_list))
            history['val_sens'].append(np.mean(sens_list))

        if epoch % 10 == 0 or epoch == cfg.stage1_epochs:
            print(f"Stage1 Epoch {epoch:3d} | train loss = {avg_loss:.4f} | val Dice = {history['val_dice'][-1]:.4f}")

    return history


def train_stage2_with_history(model, train_loader, val_loader, cfg, device):
    """Returns history dict: train_loss, val_dice, val_iou, val_sens + breakdowns"""
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.stage2_lr, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(optimizer, cfg.warmup_epochs, cfg.stage2_epochs)

    history = {
        'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_sens': [],
        'vicreg': [], 'tversky': [], 'boundary': [], 'aux': []
    }

    for epoch in range(1, cfg.stage2_epochs + 1):
        lv = 2.0 if epoch <= cfg.latent_warmup_epochs else 0.5
        criterion = SegmentationLoss(cfg, lambda_vicreg=lv)

        model.train()
        model.mask_encoder.eval()
        total_loss = 0.0
        breakdown_acc = {'vicreg': 0., 'tversky': 0., 'boundary': 0., 'aux': 0.}

        for images, masks in train_loader:
            images = images.float().to(device)
            masks  = masks.float().to(device)
            optimizer.zero_grad()
            pred_logits, aux_logits, z_pred, z_mask = model(images, masks)
            loss, bd = criterion(pred_logits, aux_logits, z_pred, z_mask, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()
            for k in breakdown_acc: breakdown_acc[k] += bd[k]

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        for k in breakdown_acc:
            history[k].append(breakdown_acc[k] / len(train_loader))

        # Validation
        if val_loader is not None:
            model.eval()
            dice_list, iou_list, sens_list = [], [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.float().to(device)
                    masks  = masks.float().to(device)
                    pred_logits = model(images)   # inference only
                    metrics = compute_metrics(pred_logits, masks)
                    dice_list.append(metrics['dice'])
                    iou_list.append(metrics['iou'])
                    sens_list.append(metrics['sensitivity'])
            history['val_dice'].append(np.mean(dice_list))
            history['val_iou'].append(np.mean(iou_list))
            history['val_sens'].append(np.mean(sens_list))

        if epoch % 10 == 0 or epoch == cfg.stage2_epochs:
            print(f"Stage2 Epoch {epoch:3d} | loss={avg_loss:.4f} | val Dice={history['val_dice'][-1]:.4f} | Sens={history['val_sens'][-1]:.4f}")

    return history
 [code] {"execution":{"iopub.status.busy":"2026-06-07T17:23:08.190186Z","iopub.execute_input":"2026-06-07T17:23:08.190499Z"}}
cfg = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Stage 1
mae = MaskAutoencoder(cfg)
history1 = train_stage1_with_history(mae, train_loader, val_loader, cfg, device)

# Stage 2
model = LatentSegNet(cfg, pretrained_mae=mae)
history2 = train_stage2_with_history(model, train_loader, val_loader, cfg, device)

# Save final model
torch.save({
    'model_state': model.state_dict(),
    'cfg': cfg,
    'history1': history1,
    'history2': history2,
}, 'latentsegnet_v2_complete.pt')
print("Training complete. Model saved.")
 [code]
import matplotlib.pyplot as plt

def plot_training_curves(history1, history2):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Stage 1
    axes[0,0].plot(history1['train_loss'], label='Train Loss')
    axes[0,0].set_title('Stage 1 (Autoencoder)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)

    # Stage 2 losses
    axes[0,1].plot(history2['train_loss'], label='Train Total Loss')
    axes[0,1].plot(history2['vicreg'], label='VICReg')
    axes[0,1].plot(history2['tversky'], label='Tversky')
    axes[0,1].plot(history2['boundary'], label='Boundary')
    axes[0,1].plot(history2['aux'], label='Aux')
    axes[0,1].set_title('Stage 2 – Loss Components')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Stage 2 validation metrics
    axes[1,0].plot(history2['val_dice'], label='Dice')
    axes[1,0].plot(history2['val_iou'], label='IoU')
    axes[1,0].plot(history2['val_sens'], label='Sensitivity')
    axes[1,0].set_title('Stage 2 – Validation Metrics')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Score')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Compare Stage1 vs Stage2 validation Dice
    axes[1,1].plot(history1['val_dice'], label='Stage1 (AE)')
    axes[1,1].plot(history2['val_dice'], label='Stage2 (Seg)')
    axes[1,1].set_title('Validation Dice: Stage1 vs Stage2')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Dice')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

plot_training_curves(history1, history2)
 [code]
def show_predictions(model, dataset, indices, device='cuda', threshold=0.5):
    """Display image, ground truth, and predicted mask for given indices."""
    model.eval()
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4*len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        img, mask = dataset[idx]          # img: 3×512×512, mask: 1×512×512
        img_disp = img.permute(1,2,0).cpu().numpy()
        mask_disp = mask.squeeze(0).cpu().numpy()

        # Predict
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            pred = (prob > threshold).astype(np.uint8)

        axes[row, 0].imshow(img_disp)
        axes[row, 0].set_title(f"Original Image (idx={idx})")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask_disp, cmap='gray')
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(pred, cmap='gray')
        axes[row, 2].set_title(f"Predicted Mask (threshold={threshold})")
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.show()

# Choose 3 samples from the validation set
val_indices = np.random.choice(len(val_ds), 3, replace=False)
show_predictions(model, val_ds, val_indices, device=device, threshold=0.5)
"""
DualEncoderSeg v3 — Latent-Space Bridging with Dual Skip Connections
=====================================================================

Core idea (your original vision, properly realised):
  Train from BOTH sides — image encoder and mask encoder each learn
  their own skip features and latent representation.
  Prediction (mask reconstruction) happens by passing the IMAGE latent
  through the MaskDecoder — the same decoder trained by the mask
  autoencoder in Stage 1.
  The decoder receives skip connections from the IMAGE encoder, not the
  mask encoder, so it uses actual image texture to refine vessel edges.

Root problem in v2 that this fixes:
  v2 MaskDecoder had NO skip connections at all — it decoded from
  a 16×16 bottleneck alone, which destroys thin vessel structure.
  The spatial information from image intermediate layers was
  completely thrown away after the pooling step.

Solution — Dual-Path Skip Injection (DPSI):
  ┌─────────────────────────────────────────────────────┐
  │ Stage 1 — Train MaskAutoencoder with self-skips     │
  │   mask -> MaskEncoder (saves s1…s4) -> latent z_m    │
  │         -> MaskDecoder (uses s4…s1) -> recon mask    │
  │   Loss: Tversky + Boundary                          │
  └─────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────┐
  │ Stage 2 — Train LatentSegNet with cross-skips       │
  │   image -> ImageEncoder (saves f1…f4) -> z_img       │
  │         -> MappingNet -> z_pred (≈ z_mask target)    │
  │         -> MaskDecoder (uses f4…f1) -> pred logits   │
  │   mask  -> MaskEncoder (frozen) -> z_mask            │
  │   Loss: VICReg(z_pred, z_mask) + Tversky + Boundary│
  └─────────────────────────────────────────────────────┘

Key changes over v2:
  1. MaskEncoder / ImageEncoder both return (latent, [s1,s2,s3,s4])
     — skip feature maps at 256×256, 128×128, 64×64, 32×32
  2. MaskDecoder accepts skip list and fuses via learned gate
     — in Stage 1: uses mask skips (self-supervised)
     — in Stage 2: uses image skips (cross-modal supervision)
  3. SkipFusionGate: learns how much of each skip to trust
     — sigmoid gate per-channel, prevents skip dominating latent
  4. Corrected VICReg: variance/covariance computed over batch B
     — spatial positions are NOT collapsed together
  5. Auxiliary head at latent resolution (16×16 -> 32×32)
     — short gradient path from loss -> MappingNet
  6. AdaptiveAvgPool removed from ImageEncoder
     — after 5 stride-2 down-stages on 512 input -> output is exactly 16×16
     — no pooling needed (pool was hiding bugs in resolution tracking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Encoder output resolution — 512 / 2^5 = 16
    spatial_h:   int   = 16
    spatial_w:   int   = 16
    latent_ch:   int   = 128       # channels in spatial latent (16×16×128)

    image_size:  int   = 512
    dropout:     float = 0.1

    # ── Loss weights ────────────────────────────────────────────────
    # VICReg: keep sim low, var/cov act as regularisers only
    lambda_vicreg_sim: float = 5.0    # was 25.0 — was dominating pixel losses
    lambda_vicreg_var: float = 5.0    # was 25.0
    lambda_vicreg_cov: float = 1.0

    lambda_tversky:  float = 1.5
    lambda_boundary: float = 1.0
    lambda_aux:      float = 0.4      # auxiliary latent-resolution head

    # Tversky: beta > alpha -> missing vessels penalised more than FP
    tversky_alpha: float = 0.3
    tversky_beta:  float = 0.7

    # ── Training schedule ───────────────────────────────────────────
    stage1_epochs:        int   = 60
    stage2_epochs:        int   = 120
    stage1_lr:            float = 1e-3
    stage2_lr:            float = 3e-4
    warmup_epochs:        int   = 10
    latent_warmup_epochs: int   = 20   # VICReg curriculum: high -> low weight


# ═══════════════════════════════════════════════════════════════════
# PRIMITIVE BLOCKS
# ═══════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class SEResBlock(nn.Module):
    """Squeeze-and-Excitation residual block — unchanged from v2."""
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
        res   = self.conv(x)
        scale = self.se(res).view(res.size(0), -1, 1, 1)
        return self.relu(x + res * scale)


class DownStage(nn.Module):
    """Stride-2 conv + SEResBlock — halves spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=2),
            SEResBlock(out_ch),
        )
    def forward(self, x): return self.block(x)


# ═══════════════════════════════════════════════════════════════════
# SKIP FUSION GATE
# ═══════════════════════════════════════════════════════════════════

class SkipFusionGate(nn.Module):
    """
    Learned gating of a skip connection before adding to decoder feature.

    Why a gate instead of plain addition?
      In Stage 1: the skip and the upsampled decoder are from the SAME
      encoder (mask -> mask), so both are in the same feature space.
      In Stage 2: the skip is from ImageEncoder, the decoder context is
      from MaskDecoder trained on masks — different domains.
      A sigmoid gate lets the decoder learn "how much of this image skip
      to trust at each channel", avoiding destructive interference.

    Gate formula:
      g = σ(W_skip · skip + W_ctx · ctx)     ← soft channel attention
      out = ctx + g ⊙ conv(skip)             ← gated residual
    """
    def __init__(self, skip_ch, ctx_ch, out_ch):
        super().__init__()
        # Project skip to decoder channel space
        self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
        # Gate: uses both skip and context to decide trust per channel
        self.gate = nn.Sequential(
            nn.Conv2d(skip_ch + ctx_ch, out_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.ctx_proj  = nn.Conv2d(ctx_ch, out_ch, 1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)

    def forward(self, skip: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        skip: (B, skip_ch, H, W) — from encoder at this resolution
        ctx:  (B, ctx_ch,  H, W) — from decoder (upsampled)
        Returns (B, out_ch, H, W)
        """
        # Align spatial size (handles off-by-one from bilinear upsample)
        if skip.shape[-2:] != ctx.shape[-2:]:
            skip = F.interpolate(skip, size=ctx.shape[-2:],
                                 mode='bilinear', align_corners=False)
        gate  = self.gate(torch.cat([skip, ctx], dim=1))   # (B, out_ch, H, W)
        fused = self.ctx_proj(ctx) + gate * self.skip_proj(skip)
        return self.bn(fused)


# ═══════════════════════════════════════════════════════════════════
# MASK ENCODER (Stage 1 teacher / Stage 2 frozen target)
# ═══════════════════════════════════════════════════════════════════

class MaskEncoder(nn.Module):
    """
    Encodes 512×512×1 binary mask -> latent (B, latent_ch, 16, 16)
    AND returns intermediate skip features.

    Skip map resolutions (after each DownStage):
      s1: B×64×256×256
      s2: B×128×128×128
      s3: B×256×64×64
      s4: B×256×32×32
    Latent z: B×latent_ch×16×16
    """
    # Channel sizes at each skip level — needed by decoder
    SKIP_CHS = [64, 128, 256, 256]   # s1 … s4

    def __init__(self, cfg: Config):
        super().__init__()
        self.stem  = ConvBNReLU(1, 32, stride=1)    # 512×512, 32 ch
        self.down1 = DownStage(32,  64)              # 256×256, 64 ch  -> s1
        self.down2 = DownStage(64,  128)             # 128×128, 128 ch -> s2
        self.down3 = DownStage(128, 256)             # 64×64,   256 ch -> s3
        self.down4 = DownStage(256, 256)             # 32×32,   256 ch -> s4
        self.down5 = DownStage(256, 256)             # 16×16,   256 ch -> latent

        self.project = nn.Sequential(
            nn.Conv2d(256, cfg.latent_ch, 1, bias=False),
            nn.BatchNorm2d(cfg.latent_ch),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
          z     : (B, latent_ch, 16, 16)
          skips : [s1, s2, s3, s4]  — high-to-low resolution
        """
        x  = self.stem(x)
        s1 = self.down1(x)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        s4 = self.down4(s3)
        x5 = self.down5(s4)
        z  = self.project(x5)
        return z, [s1, s2, s3, s4]


# ═══════════════════════════════════════════════════════════════════
# IMAGE ENCODER (Stage 2 student)
# ═══════════════════════════════════════════════════════════════════

class ImageEncoder(nn.Module):
    """
    Encodes 512×512×3 retina image -> latent (B, latent_ch, 16, 16)
    AND returns intermediate skip features at the SAME resolutions as
    MaskEncoder — critical for cross-modal skip injection.

    Architecture deliberately mirrors MaskEncoder:
      same depth, same resolution pyramid.
      Only difference: in_ch=3, initial channels 32->64.

    Skip map resolutions:
      f1: B×64×256×256
      f2: B×128×128×128
      f3: B×256×64×64
      f4: B×256×32×32
    Latent z_img: B×latent_ch×16×16

    NOTE on AdaptiveAvgPool:
      v2 applied AdaptiveAvgPool2d(16,16) AFTER down5.
      After 5 stride-2 downsamples on a 512 input the spatial size IS
      already 16×16 — the pool was a no-op that masked resolution bugs.
      We remove it here for clarity.
    """
    # Same skip channel sizes as MaskEncoder — required for weight sharing in decoder
    SKIP_CHS = [64, 128, 256, 256]

    def __init__(self, cfg: Config):
        super().__init__()
        self.stem  = ConvBNReLU(3, 64, stride=1)    # 512×512,  64 ch
        self.down1 = DownStage(64,  64)              # 256×256,  64 ch  -> f1
        self.down2 = DownStage(64,  128)             # 128×128, 128 ch  -> f2
        self.down3 = DownStage(128, 256)             # 64×64,   256 ch  -> f3
        self.down4 = DownStage(256, 256)             # 32×32,   256 ch  -> f4
        self.down5 = DownStage(256, 256)             # 16×16,   256 ch  -> latent

        self.project = nn.Sequential(
            nn.Dropout2d(cfg.dropout),
            nn.Conv2d(256, cfg.latent_ch, 1, bias=False),
            nn.BatchNorm2d(cfg.latent_ch),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns:
          z_img : (B, latent_ch, 16, 16)
          skips : [f1, f2, f3, f4]  — high-to-low resolution
        """
        x  = self.stem(x)
        f1 = self.down1(x)
        f2 = self.down2(f1)
        f3 = self.down3(f2)
        f4 = self.down4(f3)
        x5 = self.down5(f4)
        z  = self.project(x5)
        return z, [f1, f2, f3, f4]


# ═══════════════════════════════════════════════════════════════════
# MASK DECODER with Gated Skip Connections
# ═══════════════════════════════════════════════════════════════════

class MaskDecoder(nn.Module):
    """
    Reconstructs 512×512×1 mask from latent (B, latent_ch, 16, 16)
    using gated skip connections.

    SKIP INJECTION PROTOCOL:
      The decoder accepts skips from ANY encoder (mask or image).
      In Stage 1 it receives mask skips -> learns to reconstruct masks.
      In Stage 2 it receives IMAGE skips -> projects image textures into
      mask space, recovering fine vessel detail that the latent alone
      cannot provide.

    Resolution path:
      16 -> 32 -> 64 -> 128 -> 256 -> 512
       z    up1   up2   up3    up4   up5   out

    Skip injection points (from LOW to HIGH resolution):
      after up1 (32×32):   receives reversed s4 / f4
      after up2 (64×64):   receives reversed s3 / f3
      after up3 (128×128): receives reversed s2 / f2
      after up4 (256×256): receives reversed s1 / f1

    The encoder skip list [s1,s2,s3,s4] is HIGH->LOW resolution.
    The decoder consumes them in reverse: s4, s3, s2, s1.
    """
    def __init__(self, cfg: Config,
                 skip_chs: List[int] = MaskEncoder.SKIP_CHS):
        """
        skip_chs: channel counts [s1_ch, s2_ch, s3_ch, s4_ch]
                  from whichever encoder will supply skips at runtime.
        """
        super().__init__()
        lch = cfg.latent_ch

        # Expand latent
        self.expand = ConvBNReLU(lch, 256, kernel=1, padding=0)

        # Up-stages (no skip — pure decoder path)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(256, 256),
        )   # 16->32

        # After up1: fuse s4 (32×32)
        self.gate4 = SkipFusionGate(skip_chs[3], 256, 256)
        self.up2   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(256, 128),
        )   # 32->64

        # After up2: fuse s3 (64×64)
        self.gate3 = SkipFusionGate(skip_chs[2], 128, 128)
        self.up3   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(128, 128),
        )   # 64->128

        # After up3: fuse s2 (128×128)
        self.gate2 = SkipFusionGate(skip_chs[1], 128, 128)
        self.up4   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(128, 64),
        )   # 128->256

        # After up4: fuse s1 (256×256)
        self.gate1 = SkipFusionGate(skip_chs[0], 64, 64)
        self.up5   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNReLU(64, 32),
        )   # 256->512

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self,
                z:     torch.Tensor,
                skips: Optional[List[torch.Tensor]] = None,
                ) -> torch.Tensor:
        """
        z:     (B, latent_ch, 16, 16)
        skips: [s1, s2, s3, s4] HIGH->LOW resolution (or None)
        Returns logits (B, 1, 512, 512)
        """
        x = self.expand(z)          # (B, 256, 16, 16)
        x = self.up1(x)             # (B, 256, 32, 32)

        if skips is not None:
            # skips[3] = s4 (lowest resolution skip, 32×32)
            x = self.gate4(skips[3], x)

        x = self.up2(x)             # (B, 128, 64, 64)

        if skips is not None:
            x = self.gate3(skips[2], x)

        x = self.up3(x)             # (B, 128, 128, 128)

        if skips is not None:
            x = self.gate2(skips[1], x)

        x = self.up4(x)             # (B, 64, 256, 256)

        if skips is not None:
            x = self.gate1(skips[0], x)

        x = self.up5(x)             # (B, 32, 512, 512)
        return self.out(x)          # (B, 1, 512, 512) logits


# ═══════════════════════════════════════════════════════════════════
# MASK AUTOENCODER (Stage 1)
# ═══════════════════════════════════════════════════════════════════

class MaskAutoencoder(nn.Module):
    """
    Stage 1 self-supervised module.

    Forward: mask -> MaskEncoder -> (z, skips) -> MaskDecoder(z, skips) -> recon
    The decoder is trained with its own skip connections so it LEARNS
    to use skip context to reconstruct thin vessels.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = MaskEncoder(cfg)
        self.decoder = MaskDecoder(cfg, skip_chs=MaskEncoder.SKIP_CHS)

    def forward(self, mask: torch.Tensor):
        z, skips = self.encoder(mask)
        recon    = self.decoder(z, skips)
        return recon, z, skips

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        # Keep BN in eval during Stage 2
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        print("MaskEncoder frozen (weights + BN stats).")


# ═══════════════════════════════════════════════════════════════════
# LATENT MAPPING NETWORK
# ═══════════════════════════════════════════════════════════════════

class SpatialMappingNet(nn.Module):
    """
    Maps image spatial latent -> mask spatial latent.
    Both tensors: (B, latent_ch, 16, 16).

    Design:
      · 1×1 conv -> channel expansion (per-position MLP)
      · 3×3 depthwise -> spatial context (vessel continuity)
      · 1×1 conv -> project back
      · residual connection for gradient stability

    No change needed from v2 — this part was correct.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch     = cfg.latent_ch
        hidden = ch * 2

        self.net = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.GELU(),

            nn.Conv2d(hidden, hidden, 3, padding=1,
                      groups=hidden, bias=False),           # depthwise
            nn.Conv2d(hidden, hidden, 1, bias=False),       # pointwise
            nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Dropout2d(cfg.dropout),

            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.residual = nn.Conv2d(ch, ch, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z) + self.residual(z)


# ═══════════════════════════════════════════════════════════════════
# AUXILIARY HEAD (latent-resolution prediction)
# ═══════════════════════════════════════════════════════════════════

class AuxHead(nn.Module):
    """
    Predicts a 32×32 mask directly from the MAPPED latent z_pred.

    This is the "prediction at latent state" you originally wanted:
      The mapped latent IS the predicted mask representation.
      The aux head turns it into pixel logits at low resolution
      so gradients can flow back directly to MappingNet without
      waiting for all 5 decoder up-stages to contribute.

    At inference: NOT called -> zero overhead.
    At training:  contributes λ_aux × tversky(aux, downsample(gt)).
    """
    def __init__(self, cfg: Config):
        super().__init__()
        ch = cfg.latent_ch
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )   # 16->32

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_ch, 16, 16) -> logits: (B, 1, 32, 32)"""
        return self.head(z)


# ═══════════════════════════════════════════════════════════════════
# FULL SEGMENTATION MODEL (Stage 2)
# ═══════════════════════════════════════════════════════════════════

class DualEncoderSegV3(nn.Module):
    """
    Full segmentation model — the core of DualEncoderSeg v3.

    Training forward pass (mask provided):
      image -> ImageEncoder -> (z_img, [f1,f2,f3,f4])
      z_img -> MappingNet  -> z_pred  (≈ mask latent)
      z_pred -> AuxHead   -> aux_logits (32×32, short gradient path)
      z_pred + [f1..f4] -> MaskDecoder -> pred_logits (512×512)
      mask  -> MaskEncoder (frozen) -> (z_mask, _)

    Inference forward pass (mask = None):
      image -> ImageEncoder -> z_img -> MappingNet -> z_pred
      z_pred + image skips -> MaskDecoder -> logits

    The skip connections at inference come from ImageEncoder (cross-modal)
    so the decoder receives actual image features at every resolution,
    recovering fine vessel edges that are invisible in the latent alone.
    """
    def __init__(self, cfg: Config, pretrained_mae: MaskAutoencoder = None):
        super().__init__()
        self.cfg           = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.mapping       = SpatialMappingNet(cfg)
        self.aux_head      = AuxHead(cfg)

        if pretrained_mae is not None:
            self.mask_decoder = pretrained_mae.decoder
            self.mask_encoder = pretrained_mae.encoder
            pretrained_mae.freeze_encoder()
        else:
            # Cold-start: train everything end-to-end from scratch
            self.mask_encoder = MaskEncoder(cfg)
            self.mask_decoder = MaskDecoder(cfg,
                                            skip_chs=ImageEncoder.SKIP_CHS)

    def forward(self, image: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        # ── Image branch ───────────────────────────────────────────
        z_img, img_skips = self.image_encoder(image)
        z_pred           = self.mapping(z_img)

        # ── Decode with IMAGE skips (cross-modal spatial injection) ─
        pred_logits = self.mask_decoder(z_pred, img_skips)

        if mask is not None:
            # Short gradient path through latent
            aux_logits = self.aux_head(z_pred)

            # Frozen mask encoder — target latent for VICReg
            with torch.no_grad():
                z_mask, _ = self.mask_encoder(mask)

            return pred_logits, aux_logits, z_pred, z_mask

        return pred_logits


# ═══════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def tversky_loss(pred_logits: torch.Tensor,
                 target:      torch.Tensor,
                 alpha: float = 0.3,
                 beta:  float = 0.7,
                 smooth: float = 1e-5) -> torch.Tensor:
    """
    Asymmetric Dice (Tversky).
    beta > alpha -> missing vessels penalised more than false positives.
    """
    pred = torch.sigmoid(pred_logits)
    tp   = (pred * target).sum(dim=(2, 3))
    fp   = (pred * (1 - target)).sum(dim=(2, 3))
    fn   = ((1 - pred) * target).sum(dim=(2, 3))
    tv   = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tv).mean()


def boundary_loss(pred_logits: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """
    Boundary-weighted BCE.
    Applies 5× weight at vessel edges detected by Sobel on GT mask.
    """
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=target.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    edge = (
        F.conv2d(target, sobel_x, padding=1).abs() +
        F.conv2d(target, sobel_y, padding=1).abs()
    ).clamp(0, 1)
    bce      = F.binary_cross_entropy_with_logits(
        pred_logits, target, reduction='none')
    weighted = bce * (1 + 4 * edge)
    return weighted.mean()


def vicreg_loss(z_pred: torch.Tensor,
                z_mask: torch.Tensor,
                sim_w:  float = 5.0,
                var_w:  float = 5.0,
                cov_w:  float = 1.0) -> torch.Tensor:
    """
    VICReg — FIXED from v2.

    v2 bug: z was reshaped to (B*H*W, C), mixing the batch dimension
    with spatial positions. This forced ALL spatial positions to have
    the same variance/covariance distribution, which is wrong — the
    retina image has non-stationary structure (optic disc, fovea, etc.).

    Fix: variance + covariance computed over batch dimension B,
    averaged over spatial positions (H, W). Each spatial position
    maintains its own representation statistics.

    1. Invariance (sim):  MSE(z_pred, z_mask) — pull together
    2. Variance   (var):  each dim should have std ≥ 1 across batch
    3. Covariance (cov):  off-diagonal cov entries -> 0 across batch
    """
    B, C, H, W = z_pred.shape

    # 1. Invariance
    sim = F.mse_loss(z_pred, z_mask.detach())

    # 2. Variance: computed over B, averaged over spatial H,W
    #    z_pred: (B, C, H, W) -> var over dim=0 -> (C, H, W)
    std_p = torch.sqrt(z_pred.var(dim=0) + 1e-4)   # (C, H, W)
    std_m = torch.sqrt(z_mask.var(dim=0) + 1e-4)   # (C, H, W)
    var   = F.relu(1 - std_p).mean() + F.relu(1 - std_m).mean()

    # 3. Covariance: computed over B per spatial position, averaged over H,W
    #    Reshape to (H*W, B, C) for batched matrix ops
    def _off_diag_cov(z):
        zp = z - z.mean(dim=0, keepdim=True)                   # centre over B
        zf = zp.permute(2, 3, 0, 1).reshape(H * W, B, C)      # (S, B, C)
        cov = torch.bmm(zf.transpose(1, 2), zf) / max(B - 1, 1)  # (S, C, C)
        mask = 1 - torch.eye(C, device=z.device).unsqueeze(0)
        return (cov.pow(2) * mask).sum() / (C * H * W)

    cov = _off_diag_cov(z_pred) + _off_diag_cov(z_mask)

    return sim_w * sim + var_w * var + cov_w * cov


class SegmentationLoss(nn.Module):
    """
    Combined Stage 2 loss.

    Term            Weight      Purpose
    ──────────────────────────────────────────────────────────────────
    VICReg latent   λ_v         Align + stabilise latent space
    Tversky pixel   λ_t         Vessel recall bias (FN >> FP)
    Boundary pixel  λ_b         Thin vessel edge precision
    Aux 32×32       λ_a         Short gradient path -> MappingNet
    """
    def __init__(self, cfg: Config, lambda_vicreg: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.lv  = lambda_vicreg

    def forward(self, pred_logits, aux_logits, z_pred, z_mask, target_mask):
        l_vic = vicreg_loss(
            z_pred, z_mask,
            sim_w=self.cfg.lambda_vicreg_sim,
            var_w=self.cfg.lambda_vicreg_var,
            cov_w=self.cfg.lambda_vicreg_cov,
        )
        l_tvk = tversky_loss(
            pred_logits, target_mask,
            alpha=self.cfg.tversky_alpha, beta=self.cfg.tversky_beta,
        )
        l_bnd = boundary_loss(pred_logits, target_mask)

        # Auxiliary head: bilinear downsampled soft target (NOT nearest)
        aux_target = F.interpolate(
            target_mask, size=(32, 32), mode='bilinear', align_corners=False)
        l_aux = F.binary_cross_entropy_with_logits(aux_logits, aux_target)

        total = (self.lv * l_vic
                 + self.cfg.lambda_tversky  * l_tvk
                 + self.cfg.lambda_boundary * l_bnd
                 + self.cfg.lambda_aux      * l_aux)

        breakdown = {
            'vicreg': l_vic.item(), 'tversky': l_tvk.item(),
            'boundary': l_bnd.item(), 'aux': l_aux.item(),
        }
        return total, breakdown


class AutoencoderLoss(nn.Module):
    """Stage 1: Tversky + Boundary on mask reconstruction."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(self, recon_logits, target_mask):
        return (
            tversky_loss(recon_logits, target_mask,
                         alpha=self.cfg.tversky_alpha,
                         beta=self.cfg.tversky_beta)
            + boundary_loss(recon_logits, target_mask)
        )


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(pred_logits: torch.Tensor,
                    target: torch.Tensor,
                    threshold: float = 0.5) -> Dict[str, float]:
    """IoU, Dice, pixel accuracy, sensitivity, specificity."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2, 3))
    s  = 1e-5
    return {
        'iou':         ((tp + s) / (tp + fp + fn + s)).mean().item(),
        'dice':        ((2 * tp + s) / (2 * tp + fp + fn + s)).mean().item(),
        'acc':         (pred == target).float().mean().item(),
        'sensitivity': ((tp + s) / (tp + fn + s)).mean().item(),
        'specificity': ((tn + s) / (tn + fp + s)).mean().item(),
    }


# ═══════════════════════════════════════════════════════════════════
# LR SCHEDULER
# ═══════════════════════════════════════════════════════════════════

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup -> cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════
# STAGE 1 TRAINING — Mask Autoencoder
# ═══════════════════════════════════════════════════════════════════

def train_stage1(mae: MaskAutoencoder,
                 train_loader,
                 val_loader=None,
                 cfg: Config = None,
                 device: str = 'cuda') -> dict:
    """
    Pretrain the MaskAutoencoder with skip connections.
    Target: val Dice > 0.92 before proceeding to Stage 2.
    """
    if cfg is None: cfg = Config()
    mae       = mae.to(device)
    criterion = AutoencoderLoss(cfg)
    optimizer = AdamW(mae.parameters(), lr=cfg.stage1_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs)

    history = {'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_sens': []}
    best_dice = 0.0

    print("=" * 60)
    print("STAGE 1 — Mask Autoencoder (with skip connections)")
    print("=" * 60)

    for epoch in range(1, cfg.stage1_epochs + 1):
        mae.train()
        total_loss = 0.0

        for batch in train_loader:
            masks = batch[1] if isinstance(batch, (list, tuple)) else batch
            masks = masks.float().to(device)

            optimizer.zero_grad()
            recon, _, _ = mae(masks)         # returns recon, z, skips
            loss = criterion(recon, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(train_loader)
        history['train_loss'].append(avg)

        if val_loader is not None:
            mae.eval()
            dl, il, sl = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    masks = batch[1] if isinstance(batch, (list, tuple)) else batch
                    masks = masks.float().to(device)
                    recon, _, _ = mae(masks)
                    m = compute_metrics(recon, masks)
                    dl.append(m['dice'])
                    il.append(m['iou'])
                    sl.append(m['sensitivity'])
            vd, vi, vs = np.mean(dl), np.mean(il), np.mean(sl)
            history['val_dice'].append(vd)
            history['val_iou'].append(vi)
            history['val_sens'].append(vs)
            best_dice = max(best_dice, vd)

            if epoch % 10 == 0 or epoch == cfg.stage1_epochs:
                print(f"Ep {epoch:3d}/{cfg.stage1_epochs}  "
                      f"loss={avg:.4f}  Dice={vd:.4f}  "
                      f"IoU={vi:.4f}  Sens={vs:.4f}")

    if best_dice < 0.88 and val_loader is not None:
        print(f"\nWARNING:  WARNING: best val Dice={best_dice:.4f} < 0.88.")
        print("   -> Increase cfg.latent_ch or cfg.stage1_epochs.")
    print("Stage 1 complete.\n")
    return history


# ═══════════════════════════════════════════════════════════════════
# STAGE 2 TRAINING — Full Segmentation with Cross-Modal Skips
# ═══════════════════════════════════════════════════════════════════

def train_stage2(model:    DualEncoderSegV3,
                 train_loader,
                 val_loader=None,
                 cfg:      Config = None,
                 device:   str = 'cuda') -> dict:
    """
    Train the full DualEncoderSegV3.
    MaskEncoder is frozen.  ImageEncoder + MappingNet + MaskDecoder train.

    VICReg curriculum:
      Epochs 1 … latent_warmup:  lv = 2.0  (alignment dominant)
      Epochs warmup+1 …:         lv = 0.5  (pixel losses take over)
    """
    if cfg is None: cfg = Config()
    model   = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]

    print(f"Trainable params:  {sum(p.numel() for p in trainable):,}")
    print(f"Frozen params:     "
          f"{sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

    optimizer = AdamW(trainable, lr=cfg.stage2_lr, weight_decay=1e-4)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, cfg.warmup_epochs, cfg.stage2_epochs)

    history = {
        'train_loss': [], 'val_dice': [], 'val_iou': [], 'val_sens': [],
        'vicreg': [], 'tversky': [], 'boundary': [], 'aux': [],
    }

    print("=" * 60)
    print("STAGE 2 — DualEncoderSeg v3 (cross-modal skip injection)")
    print("=" * 60)

    for epoch in range(1, cfg.stage2_epochs + 1):
        lv = 2.0 if epoch <= cfg.latent_warmup_epochs else 0.5
        criterion = SegmentationLoss(cfg, lambda_vicreg=lv)

        model.train()
        # Keep frozen encoder stats fixed
        model.mask_encoder.eval()
        for m in model.mask_encoder.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        total_loss   = 0.0
        bd_acc       = {'vicreg': 0., 'tversky': 0., 'boundary': 0., 'aux': 0.}

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
            for k in bd_acc: bd_acc[k] += bd[k]

        scheduler.step()
        avg = total_loss / len(train_loader)
        history['train_loss'].append(avg)
        for k in bd_acc:
            history[k].append(bd_acc[k] / len(train_loader))

        if val_loader is not None:
            model.eval()
            dl, il, sl = [], [], []
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.float().to(device)
                    masks  = masks.float().to(device)
                    pred   = model(images)              # inference: skips from ImageEncoder
                    m      = compute_metrics(pred, masks)
                    dl.append(m['dice']); il.append(m['iou']); sl.append(m['sensitivity'])
            vd, vi, vs = np.mean(dl), np.mean(il), np.mean(sl)
            history['val_dice'].append(vd)
            history['val_iou'].append(vi)
            history['val_sens'].append(vs)

            if epoch % 10 == 0 or epoch == cfg.stage2_epochs:
                bd_str = "  ".join(
                    f"{k}={v/len(train_loader):.3f}" for k, v in bd_acc.items())
                print(f"Ep {epoch:3d}/{cfg.stage2_epochs}  "
                      f"loss={avg:.4f}  lv={lv:.1f}  "
                      f"Dice={vd:.4f}  IoU={vi:.4f}  Sens={vs:.4f}  "
                      f"[{bd_str}]")

    print("Stage 2 complete.\n")
    return history


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class RetinaDataset(torch.utils.data.Dataset):
    """
    Retina segmentation dataset.
    Augmentations applied identically to image + mask (spatial)
    and image-only (photometric).
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

        img  = Image.open(self.image_paths[idx]).convert('RGB').resize((512, 512))
        mask = Image.open(self.mask_paths[idx]).convert('L').resize((512, 512))

        if self.augment:
            # Shared spatial
            if random.random() > 0.5: img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5: img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-30, 30)
            img   = TF.rotate(img,  angle, fill=0)
            mask  = TF.rotate(mask, angle, fill=0)

            # Image-only photometric
            img = TF.adjust_brightness(img,  random.uniform(0.7, 1.3))
            img = TF.adjust_contrast(img,    random.uniform(0.7, 1.3))
            img = TF.adjust_saturation(img,  random.uniform(0.7, 1.3))
            if random.random() > 0.7:
                img = TF.gaussian_blur(img, kernel_size=3)

        img  = TF.to_tensor(img)            # 3×512×512, [0,1]
        mask = TF.to_tensor(mask)           # 1×512×512, [0,1]
        mask = (mask > 0.5).float()         # binarise
        return img, mask


# ═══════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(model: DualEncoderSegV3,
            image: torch.Tensor,
            threshold: float = 0.5,
            device: str = 'cuda') -> torch.Tensor:
    """
    Produce binary segmentation mask.
    At inference: only ImageEncoder + MappingNet + MaskDecoder are called.
    Skip connections come from the IMAGE encoder — cross-modal spatial injection.
    """
    model.eval()
    squeeze = image.dim() == 3
    if squeeze: image = image.unsqueeze(0)
    logits = model(image.to(device))
    binary = (torch.sigmoid(logits) > threshold).squeeze(1).cpu().to(torch.uint8)
    return binary.squeeze(0) if squeeze else binary


# ═══════════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════════

def sanity_check(cfg: Config = None, device: str = 'cpu'):
    if cfg is None: cfg = Config()
    print(f"Sanity check on {device}  |  "
          f"latent: {cfg.spatial_h}×{cfg.spatial_w}×{cfg.latent_ch}\n")

    B    = 2
    img  = torch.randn(B, 3, 512, 512).to(device)
    mask = torch.randint(0, 2, (B, 1, 512, 512)).float().to(device)

    # Stage 1
    mae  = MaskAutoencoder(cfg).to(device)
    recon, z, skips = mae(mask)
    assert recon.shape == (B, 1, 512, 512),      f"recon: {recon.shape}"
    assert z.shape     == (B, cfg.latent_ch, 16, 16), f"z: {z.shape}"
    assert len(skips)  == 4,                     f"skips: {len(skips)}"
    print(f"[Stage 1]  recon={tuple(recon.shape)}  z={tuple(z.shape)}  OK")
    for i, s in enumerate(skips):
        print(f"  skip[{i}]: {tuple(s.shape)}")

    # Stage 2
    mae.freeze_encoder()
    model = DualEncoderSegV3(cfg, pretrained_mae=mae).to(device)
    pred, aux, z_pred, z_mask = model(img, mask)
    assert pred.shape   == (B, 1, 512, 512),            f"pred: {pred.shape}"
    assert aux.shape    == (B, 1, 32, 32),              f"aux: {aux.shape}"
    assert z_pred.shape == (B, cfg.latent_ch, 16, 16),  f"z_pred: {z_pred.shape}"
    print(f"[Stage 2]  pred={tuple(pred.shape)}  "
          f"aux={tuple(aux.shape)}  z_pred={tuple(z_pred.shape)}  OK")

    # Loss
    criterion = SegmentationLoss(cfg, lambda_vicreg=2.0)
    loss, bd  = criterion(pred, aux, z_pred, z_mask, mask)
    assert loss.item() > 0, "Loss should be positive"
    print(f"[Loss]     total={loss.item():.4f}  breakdown={bd}  OK")

    # Inference
    binary = predict(model, img, device=device)
    assert binary.shape == (B, 512, 512), f"binary: {binary.shape}"
    print(f"[Infer]    binary={tuple(binary.shape)}  OK")
    print("\nAll checks passed.")


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def full_pipeline(train_loader,
                  val_loader=None,
                  cfg: Config = None,
                  device: str = 'cuda',
                  checkpoint_path: str = 'dual_encoder_seg_v3.pt'):
    """
    Stage 1 -> Stage 2 -> save.

    Usage:
        from pathlib import Path
        base = Path("/kaggle/input/.../Data/train")
        img_paths  = sorted((base / "image").glob("*.png"))
        mask_paths = sorted((base / "mask").glob("*.png"))
        split = int(0.8 * len(img_paths))
        from torch.utils.data import DataLoader
        train_dl = DataLoader(
            RetinaDataset(img_paths[:split], mask_paths[:split], augment=True),
            batch_size=4, shuffle=True, num_workers=2)
        val_dl = DataLoader(
            RetinaDataset(img_paths[split:], mask_paths[split:], augment=False),
            batch_size=4, shuffle=False, num_workers=2)
        model = full_pipeline(train_dl, val_dl, device='cuda')
    """
    if cfg is None: cfg = Config()

    mae     = MaskAutoencoder(cfg)
    hist1   = train_stage1(mae, train_loader, val_loader, cfg, device)

    model   = DualEncoderSegV3(cfg, pretrained_mae=mae)
    hist2   = train_stage2(model, train_loader, val_loader, cfg, device)

    torch.save({
        'model_state': model.state_dict(),
        'cfg':         cfg,
        'history1':    hist1,
        'history2':    hist2,
    }, checkpoint_path)
    print(f"Checkpoint saved -> {checkpoint_path}")
    return model


# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    sanity_check(Config(), device='cpu')

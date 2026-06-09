# Notebook Plan: DualEncoderSeg v3 vs UNet Baseline Comparison

## Objective
A single `.ipynb` file that:
- Trains both **DualEncoderSegV3** and a **UNet baseline** with matched config
- Records training histories for both
- Visualises and compares metrics, loss curves, and prediction overlays
- Is fully modular: changing one config cell re-runs everything consistently

---

## Cell-by-Cell Structure

### Section 0 — Environment & Imports
- `Cell 0.1` — pip installs (torch, torchvision, Pillow, matplotlib, seaborn)
- `Cell 0.2` — standard imports (torch, nn, F, numpy, matplotlib, pathlib, etc.)

### Section 1 — Global Config (THE single cell to change)
- `Cell 1.1` — `ExperimentConfig` dataclass
  - image_size, latent_ch, batch_size, device, seed
  - stage1_epochs, stage2_epochs, unet_epochs
  - learning rates, loss weights
  - DATA_ROOT path

### Section 2 — Model Specifications (read-only reference cell)
- `Cell 2.1` — Print model specs table:
  - Encoder type: custom SEResBlock CNN
  - Convolution: stride-2 DownStage (no MaxPool)
  - Skip mechanism: SkipFusionGate (learned gated fusion)
  - Latent: spatial 16×16×128
  - UNet baseline: standard double-conv + MaxPool + transposed conv

### Section 3 — Shared Primitive Blocks
- `Cell 3.1` — `ConvBNReLU`, `SEResBlock`, `DownStage`
- `Cell 3.2` — `SkipFusionGate`

### Section 4 — DualEncoderSeg v3 (import or inline)
- `Cell 4.1` — `MaskEncoder`, `ImageEncoder`
- `Cell 4.2` — `MaskDecoder`
- `Cell 4.3` — `MaskAutoencoder`
- `Cell 4.4` — `SpatialMappingNet`, `AuxHead`
- `Cell 4.5` — `DualEncoderSegV3`

### Section 5 — UNet Baseline (matched depth)
- `Cell 5.1` — `DoubleConv` block (Conv3×3 → BN → ReLU × 2)
- `Cell 5.2` — `UNetBaseline` class
  - 5 encoder levels matching DualEnc depth (64→128→256→512→512)
  - Max-pool downsampling
  - Transposed conv upsampling
  - Standard concat skip connections
  - Output: 1ch logits at 512×512
- `Cell 5.3` — Model spec printout comparing both

### Section 6 — Shared Loss Functions
- `Cell 6.1` — `tversky_loss`, `boundary_loss`
- `Cell 6.2` — `SegmentationLoss` (for DualEnc, includes VICReg + Aux)
- `Cell 6.3` — `UNetLoss` (Tversky + Boundary only, same λ weights)

### Section 7 — Metrics
- `Cell 7.1` — `compute_metrics` (IoU, Dice, Sensitivity, Specificity, Accuracy)

### Section 8 — Dataset & DataLoader
- `Cell 8.1` — `RetinaDataset` class (same as v3, with augmentations)
- `Cell 8.2` — DataLoader instantiation

### Section 9 — Training Functions
- `Cell 9.1` — `train_stage1` (MaskAutoencoder pre-training)
- `Cell 9.2` — `train_stage2` (DualEncoderSegV3 full training)
- `Cell 9.3` — `train_unet` (standard single-stage UNet training)
- `Cell 9.4` — `get_warmup_cosine_scheduler`

### Section 10 — Run Training
- `Cell 10.1` — Train Stage 1 (MAE)
- `Cell 10.2` — Train Stage 2 (DualEncoderSegV3)
- `Cell 10.3` — Train UNet baseline
- `Cell 10.4` — Save both model checkpoints

### Section 11 — Results & Comparison
- `Cell 11.1` — Summary metrics table (Dice, IoU, Sensitivity, Specificity)
- `Cell 11.2` — Loss curves side-by-side (DualEnc Stage 2 vs UNet)
- `Cell 11.3` — Metric curves over epochs (val Dice, IoU)
- `Cell 11.4` — Loss breakdown (DualEnc: VICReg, Tversky, Boundary, Aux)
- `Cell 11.5` — Visual overlay comparison (image | GT | DualEnc pred | UNet pred)
- `Cell 11.6` — Radar chart: 5-metric comparison

### Section 12 — Inference Utilities
- `Cell 12.1` — `predict_single` and `predict_batch` helpers
- `Cell 12.2` — Threshold sweep (Dice vs threshold for both models)

---

## Model Specification Table (printed in notebook)

| Spec                  | DualEncoderSegV3             | UNet Baseline              |
|-----------------------|------------------------------|----------------------------|
| Encoder type          | Custom SEResBlock CNN        | Standard double-conv CNN   |
| Down-sampling         | Stride-2 Conv (no MaxPool)   | MaxPool 2×2                |
| Encoder depth         | 5 stages (512→16)            | 5 stages (512→16)          |
| Skip mechanism        | SkipFusionGate (learned gate)| Standard concat + conv     |
| Latent representation | Spatial 16×16×128            | Bottleneck 16×16×512       |
| Decoder               | Bilinear upsample + ConvBN   | Transposed Conv            |
| Training stages       | 2 (MAE pretrain → finetune)  | 1 (end-to-end)             |
| Loss                  | VICReg + Tversky + Boundary + Aux | Tversky + Boundary    |
| Params (approx)       | ~8M trainable Stage 2        | ~7.7M                      |
| Aux head              | Yes (32×32 latent prediction)| No                         |

---

## Comparison Plots Plan

1. **Training Loss Curves** — 2 subplots (Stage2 DualEnc, UNet), same y-scale
2. **Val Dice over Epochs** — Single plot, two lines, shaded region for std
3. **Val IoU over Epochs** — Same format
4. **Loss Breakdown** — Stacked area chart for DualEnc (VICReg, Tversky, Boundary, Aux)
5. **Final Metric Bar Chart** — Grouped bars for all 5 metrics
6. **Radar Chart** — Pentagon comparing Dice, IoU, Sensitivity, Specificity, Accuracy
7. **Visual Prediction Grid** — 4 rows: image, GT, DualEnc, UNet (side-by-side)
8. **Threshold Sweep** — Dice vs threshold [0.1 → 0.9] for both models

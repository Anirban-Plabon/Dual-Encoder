# DualEncoderSeg v2: Architectural Overview

This document provides a comprehensive, three-level architectural overview of the **DualEncoderSeg v2** model compared to the **UNet Baseline**, as implemented in [dual_encoder_v2.ipynb](file:///F:/Anirban_250509/Projects/Dual%20Encoder/DualEncoderSeg/dual_encoder_v2.ipynb).

---

## 1. Architectural Paradigms (Level 1)

DualEncoderSeg v2 is trained using a **two-stage training protocol**, separating structural shape priors from cross-modal image-to-mask mapping.

### Stage 1: Mask Autoencoder Pretraining (Self-Supervision)
In Stage 1, the model learns a dense representation of binary vessel shapes. The encoder compresses the mask into a latent space, and the decoder reconstructs it using native mask skip connections.
```mermaid
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef stage1 fill:#d1fae5,stroke:#10b981,color:#065f46;
    
    subgraph Stage1["Stage 1: Self-Supervised Mask Autoencoder (Pretraining)"]
        M[Input Binary Mask<br>512×512×1] --> ME[Mask Encoder<br>Trainable]
        ME -->|Latent z_mask| ZM[z_mask<br>16×16×256]
        ME -->|Skips f1..f4| MS[Mask Skips<br>64, 128, 256, 512 ch]
        
        ZM --> MD[Mask Decoder<br>Trainable]
        MS -->|Direct Injection| MD
        MD --> R[Reconstructed Mask Logits<br>512×512×1]
        
        R & M --> L1[Pixel Loss Suite<br>Tversky + Boundary + Focal + Lovász]
    end
    class Stage1 stage1;
```

### Stage 2: Dual Encoder Training & Inference
In Stage 2, the Mask Encoder and Mask Decoder (upsampling layers) are frozen. An Image Encoder processes the RGB image, maps it into the mask latent space via a Spatial Mapping Network, and decodes it. Gates are unfrozen to selectively translate RGB features into the mask domain.
```mermaid
graph LR
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef stage2 fill:#dbeafe,stroke:#3b82f6,color:#1e40af;
    classDef frozen fill:#f1f5f9,stroke:#94a3b8,color:#475569;
    
    subgraph Stage2["Stage 2: Cross-Modal Segmentation (Fine-tuning & Inference)"]
        IMG[Input RGB Image<br>512×512×3] --> IE[Image Encoder<br>Trainable]
        IE -->|z_img| SMN[Spatial Mapping Net<br>Trainable]
        IE -->|RGB Skips f1..f4| G[SkipFusionGates 1..4<br>Trainable]
        
        SMN -->|z_pred| ZP[z_pred<br>16×16×256]
        
        ZP --> MD2[Mask Decoder upsampling<br>FROZEN]
        G -->|Selective Gate Projection| MD2
        
        MD2 --> OUT[Prediction Logits<br>512×512×1]
        
        %% Training Only Elements
        subgraph TrainingOnly["Supervision (Training Only)"]
            MASK[True Mask<br>512×512×1] --> ME2[Mask Encoder<br>FROZEN]
            ME2 -->|z_mask| ZM2[z_mask<br>16×16×256]
            ZP --> AUX[Aux Head<br>Trainable]
            AUX --> AUX_OUT[Aux Logits<br>32×32×1]
        end
        
        %% Losses
        OUT -->|Pixel Loss| L2[DualEncoder Loss]
        AUX_OUT -->|Aux Lovász| L2
        ZP & ZM2 -->|Latent Align: MSE + Cosine| L2
    end
    
    class Stage2 stage2;
    class ME2,MD2 frozen;
```

---

## 2. Component Pipeline & Tensor Shapes (Level 2)

This diagram details the tensor shapes, spatial dimensions, and channel counts flowing through all major sub-modules during Stage 2 training.

```mermaid
graph TB
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef tensor fill:#fef3c7,stroke:#f59e0b,color:#78350f;
    classDef frozen fill:#f1f5f9,stroke:#94a3b8,color:#475569;
    
    %% Inputs
    I("Image (B, 3, 512, 512)"):::tensor
    M("Mask (B, 1, 512, 512)"):::tensor

    %% Image Encoder
    subgraph ImageEncoder["ImageEncoder (Trainable)"]
        I_stem["Stem (DoubleConv)"] -->|"64ch | 512×512"| I_d1["DownBlock 1"]
        I_d1 -->|"64ch | 256×256"| I_d2["DownBlock 2"]
        I_d2 -->|"128ch | 128×128"| I_d3["DownBlock 3"]
        I_d3 -->|"256ch | 64×64"| I_d4["DownBlock 4"]
        I_d4 -->|"512ch | 32×32"| I_d5["DownBlock 5"]
        I_d5 -->|"512ch | 16×16"| I_proj["Dropout2d + Conv1×1"]
    end
    
    %% Skips
    I_d1 -->|f1| G1["SkipFusionGate 1"]
    I_d2 -->|f2| G2["SkipFusionGate 2"]
    I_d3 -->|f3| G3["SkipFusionGate 3"]
    I_d4 -->|f4| G4["SkipFusionGate 4"]

    %% Latent Mapping
    I_proj -->|"z_img (B, 256, 16, 16)"| SMN["SpatialMappingNet"]
    SMN -->|"z_pred (B, 256, 16, 16)"| ZP["z_pred (B, 256, 16, 16)"]:::tensor

    %% Mask Decoder
    subgraph MaskDecoder["MaskDecoder (Selective Freeze)"]
        EXP["Expand Conv (FROZEN)"] -->|"512ch | 16×16"| U1["UpBlock 1 (FROZEN)"]
        U1 -->|"512ch | 32×32"| G4
        G4 -->|"512ch | 32×32"| U2["UpBlock 2 (FROZEN)"]
        U2 -->|"256ch | 64×64"| G3
        G3 -->|"256ch | 64×64"| U3["UpBlock 3 (FROZEN)"]
        U3 -->|"128ch | 128×128"| G2
        G2 -->|"128ch | 128×128"| U4["UpBlock 4 (FROZEN)"]
        U4 -->|"64ch | 256×256"| G1
        G1 -->|"64ch | 256×256"| U5["UpBlock 5 (FROZEN)"]
        U5 -->|"32ch | 512×512"| OUT_CONV["Conv1×1 (FROZEN)"]
    end
    
    ZP --> EXP
    OUT_CONV -->|"Logits (B, 1, 512, 512)"| PL["Pixel Loss Evaluation"]
    
    %% Mask Encoder
    subgraph MaskEncoder["MaskEncoder (FROZEN)"]
        M --> M_stem["Stem"] --> M_d1["Down 1"] --> M_d2["Down 2"] --> M_d3["Down 3"] --> M_d4["Down 4"] --> M_d5["Down 5"] --> M_proj["Proj"]
    end
    class MaskEncoder,EXP,U1,U2,U3,U4,U5,OUT_CONV frozen;
    
    M_proj -->|"z_mask (B, 256, 16, 16)"| ZM["z_mask (B, 256, 16, 16)"]:::tensor
    ZP & ZM -->|"MSE + Cosine Distance"| AL["Latent Alignment Loss"]
    
    %% Aux Head
    ZP --> AUX["Aux Head"]
    AUX -->|"Aux Logits (B, 1, 32, 32)"| AUX_LOSS["Aux Lovász-Hinge Loss"]
```

---

## 3. Micro-Architecture Details (Level 3)

This section maps out the exact layer progressions and operations inside the primitives.

### Block Primitives
```mermaid
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef op fill:#eff6ff,stroke:#3b82f6,color:#1e40af;
    
    subgraph DoubleConv["DoubleConv (Atomic Block)"]
        DC_in([Input Tensor]) --> DC_c1["Conv3×3 (bias=False)"]:::op
        DC_c1 --> DC_b1["BatchNorm2d"]:::op
        DC_b1 --> DC_r1["ReLU (inplace)"]:::op
        DC_r1 --> DC_c2["Conv3×3 (bias=False)"]:::op
        DC_c2 --> DC_b2["BatchNorm2d"]:::op
        DC_b2 --> DC_r2["ReLU (inplace)"]:::op
        DC_r2 --> DC_out([Output Tensor])
    end
```

### SkipFusionGate Gating Mechanism
A learned channel-wise gate dynamically controls how much RGB information passes into the mask domain.
$$\mathbf{g} = \sigma(\mathbf{W} \cdot [\mathbf{x}_{\text{skip}}, \mathbf{x}_{\text{ctx}}])$$
$$\mathbf{x}_{\text{out}} = \text{BN}(\text{proj}_{\text{ctx}}(\mathbf{x}_{\text{ctx}}) + \mathbf{g} \odot \text{proj}_{\text{skip}}(\mathbf{x}_{\text{skip}}))$$
```mermaid
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef op fill:#eff6ff,stroke:#3b82f6,color:#1e40af;
    
    subgraph SFG["SkipFusionGate"]
        S[skip: B, skip_ch, H_s, W_s] -->|Bilinear Interpolate if shapes mismatch| S_aligned[skip: B, skip_ch, H, W]
        C[ctx: B, ctx_ch, H, W] --> CAT["Concat (dim=1)"]:::op
        S_aligned --> CAT
        CAT --> GATE_CONV["Conv2d (1×1, bias=True)"]:::op
        GATE_CONV --> SIG["Sigmoid"]:::op
        SIG -->|Gate vector 'g'| G_MUL["Elementwise Mul (⊙)"]:::op
        
        S_aligned --> S_PROJ["Conv2d (1×1, bias=False)"]:::op
        S_PROJ --> G_MUL
        
        C --> C_PROJ["Conv2d (1×1, bias=False)"]:::op
        C_PROJ --> ADD["Elementwise Add (+)"]:::op
        G_MUL --> ADD
        ADD --> BN["BatchNorm2d"]:::op
        BN --> OUT[out: B, out_ch, H, W]
    end
```

### Latent Space Bridge (Mapping Bridge Options)
The latent bridge maps the image representation into the target mask representation. There are two options supported:

1. **Option 1: Spatial Conv Bridge (`SpatialMappingNet`)** - Preserves local spatial alignment using a depthwise-separable bottleneck with a residual shortcut. Highly parameter-efficient.
2. **Option 2: True Flattened Latent DNN (`GlobalLatentDNN`)** - Flattens the $16 \times 16$ grid to allow global spatial communication across all coordinates through deep dense layers (MLP).

#### Option 1: SpatialMappingNet (Local Conv)
```mermaid
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef op fill:#eff6ff,stroke:#3b82f6,color:#1e40af;
    
    subgraph SMN["SpatialMappingNet"]
        SMN_in[z_img: B, 256, 16, 16] --> SMN_expand["Conv1×1 (256→512, bias=False)"]:::op
        SMN_expand --> SMN_bn1["BatchNorm2d"]:::op
        SMN_bn1 --> SMN_a1["GELU"]:::op
        
        SMN_a1 --> SMN_dw["Depthwise Conv3×3 (groups=512, padding=1, bias=False)"]:::op
        SMN_dw --> SMN_pw["Pointwise Conv1×1 (512→512, bias=False)"]:::op
        
        SMN_pw --> SMN_bn2["BatchNorm2d"]:::op
        SMN_bn2 --> SMN_a2["GELU"]:::op
        SMN_a2 --> SMN_drop["Dropout2d (p=0.1)"]:::op
        SMN_drop --> SMN_proj["Conv1×1 (512→256, bias=False)"]:::op
        SMN_proj --> SMN_bn3["BatchNorm2d"]:::op
        
        SMN_in --> SMN_res["Conv1×1 (256→256, bias=False)"]:::op
        
        SMN_bn3 --> SMN_add["Add (+)"]:::op
        SMN_res --> SMN_add
        
        SMN_add --> SMN_out[z_pred: B, 256, 16, 16]
    end
```

#### Option 2: GlobalLatentDNN (Global MLP)
```mermaid
graph TD
    classDef default fill:#f8fafc,stroke:#cbd5e1,color:#0f172a;
    classDef op fill:#eff6ff,stroke:#3b82f6,color:#1e40af;
    
    subgraph GDNN["GlobalLatentDNN"]
        GDNN_in[x: B, 256, 16, 16] --> GDNN_flat["Flatten (x.view(B, -1))"]:::op
        GDNN_flat -->|"B, 65536"| GDNN_l1["Linear (65536→1024)"]:::op
        GDNN_l1 --> GDNN_ln1["LayerNorm"]:::op
        GDNN_ln1 --> GDNN_a1["GELU"]:::op
        GDNN_a1 --> GDNN_d1["Dropout (p=0.1)"]:::op
        
        GDNN_d1 --> GDNN_l2["Linear (1024→1024)"]:::op
        GDNN_l2 --> GDNN_ln2["LayerNorm"]:::op
        GDNN_ln2 --> GDNN_a2["GELU"]:::op
        GDNN_a2 --> GDNN_d2["Dropout (p=0.1)"]:::op
        
        GDNN_d2 --> GDNN_l3["Linear (1024→65536)"]:::op
        GDNN_l3 --> GDNN_ln3["LayerNorm"]:::op
        
        GDNN_in --> GDNN_add["Add (+) / Skip Connection"]:::op
        GDNN_ln3 -->|"Reshape back (B, 256, 16, 16)"| GDNN_add
        GDNN_add --> GDNN_out[z_pred: B, 256, 16, 16]
    end
```

---

## 4. Controlled Variables & Comparison Matrix

To ensure a fair comparison during evaluation, components are aligned as follows:

| Feature / Param | DualEncoderSeg v2 | UNet Baseline | Alignment Logic |
|---|---|---|---|
| **Image Resolution** | $512 \times 512$ | $512 \times 512$ | Identical receptive fields |
| **Conv Block** | `DoubleConv` (no attention) | `DoubleConv` | Eliminates block-level bias |
| **Downsampling** | `MaxPool2d` | `MaxPool2d` | Identical spatial collapse dynamics |
| **Channel Progression** | $64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 512$ | $64 \rightarrow 128 \rightarrow 256 \rightarrow 512 \rightarrow 512$ | Controlled capacity |
| **Latent Dimension** | $16 \times 16 \times 256$ | $16 \times 16 \times 512$ | DualEncoder projects to tighter bottleneck |
| **Mapping Bridge** | `SpatialMappingNet` OR `GlobalLatentDNN` (configurable) | None | Controlled variable: testing mapped bridge mapping vs unmapped bottleneck |
| **Skip Connections** | Gated (`SkipFusionGate`) | Concatenation + DoubleConv | Controlled variable: testing gating efficacy |
| **Parameter Budget** | **~13.1M** (spatial) or **~146.5M** (global_dnn) | **~13.3M** | Controlled capacity boundary |

---

## 5. Loss Suite & Training Protocol

### Combined Pixel Loss (Shared)
$$\mathcal{L}_{\text{pixel}} = \lambda_{\text{tvk}}\mathcal{L}_{\text{Tversky}} + \lambda_{\text{bnd}}\mathcal{L}_{\text{Boundary}} + \lambda_{\text{fcl}}\mathcal{L}_{\text{Focal}} + \lambda_{\text{lov}}\mathcal{L}_{\text{Lov\acute{a}sz}}$$

- **Tversky ($\alpha=0.3, \beta=0.7$)**: Focuses on vessel recall (FN penalized higher).
- **Boundary**: Sobel-weighted BCE that applies $5\times$ penalty at edges.
- **Focal ($\gamma=2.0$)**: Down-weights easy background pixels.
- **Lovász-Hinge**: Smooth mathematical surrogate directly optimizing intersection-over-union.

### DualEncoder Alignment Loss (Stage 2 Only)
In Stage 2, the total objective is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pixel}} + \lambda_{\text{align}}(t) \mathcal{L}_{\text{align}}(\mathbf{z}_{\text{pred}}, \mathbf{z}_{\text{mask}}) + \lambda_{\text{aux}}\mathcal{L}_{\text{Lov\acute{a}sz-Aux}}$$

- **Latent Alignment Loss ($\mathcal{L}_{\text{align}}$)**:
  $$\mathcal{L}_{\text{align}} = \text{MSE}(\mathbf{z}_{\text{pred}}, \mathbf{z}_{\text{mask}}) + \left(1 - \text{CosineSimilarity}(\mathbf{z}_{\text{pred}}, \mathbf{z}_{\text{mask}})\right)$$
- **Alignment Weight Curriculum ($\lambda_{\text{align}}(t)$)**:
  - **Epochs 1–20**: $1.0$ (Forces coarse structural alignment).
  - **Epochs 21–60**: $0.5$ (Transitions control to segmentor).
  - **Epochs 61+**: $0.2$ (Anchors latent representation while maximizing boundary metrics).
- **Auxiliary Supervision ($\mathcal{L}_{\text{Lov\acute{a}sz-Aux}}$)**: Evaluated at $32 \times 32$ via the `AuxHead` to enforce correct shape prediction directly at the latent mapped bridge.

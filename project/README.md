# D7047E ADL — Cross-out Detection in Handwritten Documents
**Group 14 · Luleå University of Technology · 2026**

> Suresh Balaraman · Nagarajan Ganesan · Chirag Gurumurthy · Shameena Mohammed Nabeel · Lakshmi Shankar Paramasivan

**Repository:** https://github.com/chirag-gurumurthy/D7047E_ADL/tree/project  
**WandB Project:** `naggan-4-lule-university-of-technology/adl-crossout-v5`

---

## What This Project Does

Handwritten documents often contain cross-outs — lines, scribbles or marks drawn over words to delete them. These artifacts confuse automated text recognition systems. This project builds a deep learning pipeline that:

1. **Detects** whether a word image is clean or crossed-out (binary classification)
2. **Identifies** which of 7 cross-out styles was used (multi-class classification)
3. **Tests generalisation** on real hand-drawn images beyond the training distribution

---

## Dataset

Source: IAM Handwriting Database with synthetic cross-outs added on top.

~48,000 word images per class, pre-split into train / val / test.  
Image size: ~68×136 px (median H×W), grayscale, black ink on white background.

| Category | Description |
|---|---|
| CLEAN | no cross-out (baseline) |
| SINGLE_LINE | one horizontal line through the word |
| DOUBLE_LINE | two parallel horizontal lines |
| DIAGONAL | one diagonal line |
| CROSS | two crossing lines forming an X |
| WAVE | a wavy horizontal line |
| ZIG_ZAG | zig-zag line |
| SCRATCH | heavy scribbling covering the word |

`MIXED` = randomly chosen style per image — used only in binary Task 1 (it is crossed-out but has no classifiable style for Task 2).

**Paired dataset:** the exact same word image appears in every category folder. `a01-000u-00-00.png` in `CLEAN/` and in `SINGLE_LINE/` is the same word with the cross-out drawn on top. Every style was applied to the same set of words — a controlled experiment.

**Splits:** train 431,973 / val 68,031 / test 182,754 (across all classes, binary)

---

## Preprocessing Pipeline

Every image goes through the same pipeline before the model sees it:

| Step | Transform | Why |
|---|---|---|
| 1 | Grayscale → 3-channel (replicate) | Pretrained models expect 3-channel input; no colour information is added |
| 2 | PadToSquare (fill=255 white) | Word images are wide and short; direct resize would stretch and distort cross-out geometry (a diagonal would no longer look diagonal) |
| 3 | Resize to 224×224 | Required input size for ResNet-50, EfficientNet-B0, and ViT-B/16 |
| 4 | ToTensor | Converts PIL image (0–255 integers) to PyTorch float tensor (0.0–1.0) |
| 5 | Normalize (ImageNet stats) | Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225]; keeps input distribution consistent with pretrained weights |

**Training augmentation only** (not applied to val/test/Custom-50):

| Augmentation | What it simulates |
|---|---|
| RandomRotation ±5° | Tilted paper or camera |
| RandomAffine (5% shift) | Slightly off-centre capture |
| RandomPerspective (p=0.5) | Angled photo |
| GaussianBlur σ∈[0.1,1.0] | Camera blur / low resolution |
| ColorJitter ±0.3 | Variable lighting |

`RandomHorizontalFlip` is NOT used — flipping changes the visual meaning of cross-out styles (a diagonal would mirror direction).

---

## Architecture

### SimpleCNN (trained from scratch)

7 convolutional layers in 4 blocks:

```
Input (3 × 224 × 224)
  Block 1: Conv(3→32)×2  + BN + ReLU + MaxPool  →  32 × 112 × 112
  Block 2: Conv(32→64)×2 + BN + ReLU + MaxPool  →  64 × 56 × 56
  Block 3: Conv(64→128)  + Conv(128→128) + BN    →  128 × 56 × 56  [no pool]
  Block 4: Conv(128→256) + BN + AdaptiveAvgPool  →  256 × 7 × 7
  Flatten → Linear(12544→512) → ReLU → Dropout(0.4)
          → Linear(512→128)   → ReLU → Dropout(0.3)
          → Linear(128→N)     [N=2 binary, N=8 multiclass]
```

Blocks 3–4 deliberately skip pooling to preserve spatial detail needed to distinguish fine-grained patterns like Wave vs Zig-Zag. BatchNorm after every conv stabilises training.

**Trainable parameters:** 7.07M

**What each component does:**
- **Conv2d** — a filter (small grid of learned numbers) slides across the image and computes a dot product at each position, detecting local patterns: edges, lines, curves. With 32 filters in layer 1, it learns 32 different pattern detectors simultaneously.
- **BatchNorm** — normalises activations after each layer, making training more stable and acting as regularisation.
- **ReLU** — sets all negative values to 0; introduces non-linearity (without it, stacking layers would be equivalent to a single linear transformation).
- **MaxPool** — takes the maximum value in each 2×2 region, halving the spatial size; makes features slightly position-invariant and reduces computation.
- **AdaptiveAvgPool** — resizes the feature map to a fixed output size regardless of input dimensions.
- **Dropout** — during training, randomly zeros a fraction of activations; forces the network not to rely on any single neuron (reduces overfitting).

### ResNet-50 (transfer learning — multiclass)

Pre-trained on ImageNet. `layer4` (the final residual block) and the classification head are unfrozen:

```
Frozen: conv1 → layer1 → layer2 → layer3     [~17M params, fixed]
Unfrozen: layer4                               [~8.4M params, fine-tuned]
New head: Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→8)
```

**Residual connections** (what makes ResNet special):
```
Standard CNN:  x → Conv → Conv → output
ResNet:        x → Conv → Conv → output + x   (skip connection)
```
The `+ x` (adding the input directly to the output) solves the vanishing gradient problem in very deep networks. Gradients can flow back through the skip connection without passing through all conv layers, enabling training of 50+ layer networks.

Unfreezing `layer4` lets the model adapt high-level features (shapes, stroke patterns) specific to cross-outs, while the frozen lower layers retain universal edge/texture detectors.

### ViT-B/16 (exploratory — binary only)

No convolutions at all. Architecture:
1. **Patch embedding** — split the 224×224 image into 196 patches of 16×16 px each; each patch (768 values) is projected to a 768-d vector.
2. **Positional encoding** — learned position embedding added to each patch vector so the Transformer knows spatial order.
3. **Transformer encoder (12 layers)** — multi-head self-attention lets each patch "look at" all other patches weighted by relevance; captures global relationships (a cross-out line at the top influences representations at the bottom).
4. **CLS token** — special learnable token prepended to the sequence; after 12 layers, its output is used for classification via `Linear(768→2)`.

Cross-out marks span the entire word — ViT's global attention from layer 1 is potentially better at capturing long horizontal or diagonal lines than a CNN that builds global context gradually. Used as exploratory comparison only.

---

## Key Concepts

### Class Imbalance
The binary dataset has 8× more crossed-out images than clean ones (1:8 ratio). If ignored, the model learns to always predict "crossed-out" and achieves 88% accuracy without learning anything useful.

**Fix:** weighted cross-entropy loss — the CLEAN class is given 8× more weight so the model is penalised heavily for missing it.

### Transfer Learning
Instead of training from scratch on a small dataset, we start with a model already trained on millions of ImageNet images and adapt it to our task.

- **Frozen backbone** — pretrained layers are kept fixed; only the new classification head trains. Fast and avoids overfitting.
- **Partially unfrozen** — for multiclass, ResNet-50's `layer4` is unfrozen to allow fine-grained feature adaptation.
- **Why it works** — early CNN layers (edges, textures) are universal; only later layers need to be task-specific.

### Learning Rate Scheduling — ReduceLROnPlateau
Monitors validation loss. If it stops improving for `patience=5` epochs, the LR is multiplied by `factor=0.1`:

```
Initial LR → (5 epochs no improvement) → LR × 0.1 → (5 more) → LR × 0.1 → min_lr
```

For SimpleCNN: `1e-4 → 1e-5 → 1e-6`  
For ResNet-50: `4e-4 → 4e-5 → 4e-6 → 1e-6` (clamped to min_lr)

### Early Stopping
Training stops automatically when the model stops improving, preventing overfitting and saving compute.

- `patience=15` — stop if val loss does not improve for 15 consecutive epochs
- `min_epochs=20` — never stop before 20 epochs (allow the model to warm up first)
- The **best checkpoint** (lowest val loss) is saved and reloaded at the end

### Evaluation Metrics

- **Accuracy** — fraction of correctly classified samples. Can be misleading with class imbalance.
- **Precision** — of all samples predicted as class X, how many actually were X? (avoids false alarms)
- **Recall** — of all actual class X samples, how many did the model find? (avoids missing them)
- **F1 Score** — harmonic mean of Precision and Recall: `2 × P × R / (P + R)`
- **Macro-F1** — average F1 across all classes, treating each class equally regardless of size
- **Generalisation Gap** — `IAM test F1 − Custom-50 F1`. Measures performance drop on real unseen data.

### Domain Shift
IAM images are clean digital scans; Custom-50 images are real photographs. This mismatch causes performance drops:
- **Binary:** IAM F1 = 0.991, Custom-50 F1 = 0.932 → gap of 0.059 (acceptable)
- **Multiclass (prev. run):** IAM F1 ≈ 0.909, Custom-50 F1 ≈ 0.446 → gap of ~0.46 (significant)

The augmentation pipeline (perspective, blur, jitter) is specifically designed to partially bridge this gap.

### Why Binary Outperforms Multiclass (F1 0.991 vs 0.913)

The performance gap is partly explained by task difficulty (2-class vs 8-class), but the **MIXED class** likely plays a role too:

- **Binary training includes MIXED** (47,997 samples) — these are images where a randomly-chosen cross-out style was applied. The binary model therefore sees extra variety in the positive class: ambiguous strokes, transitional patterns, and style combinations that don't fit neatly into any one category. This diversity makes the binary classifier more robust.
- **Multiclass excludes MIXED** — it cannot be used because MIXED has no fixed style label to assign. The multiclass model never sees these ambiguous examples during training.

In effect, MIXED acts as a free diversity boost for binary but is unavailable to multiclass. This is one reason binary generalises more strongly — the positive class is richer and more varied. A potential future improvement for multiclass would be **synthetic augmentation** that generates similar in-between style combinations at training time.

---

## Results Summary

### Task 1 — Binary Classification (SimpleCNN, final model)

| Metric | IAM Test Set | Custom-50 |
|---|---|---|
| Val Accuracy | 0.9808 | — |
| F1 Score | 0.9908 | 0.9320 |
| Generalisation Gap | — | 0.0587 |

SimpleCNN converged within ~5 epochs and matched all pre-trained models. Selected for its speed, simplicity, and strong IAM performance.

### Task 2 — Multi-class Classification

| Model | Val Acc | Test Acc | Macro-F1 | Macro-Precision | Macro-Recall |
|---|---|---|---|---|---|
| ResNet-50 | 0.8942 | 0.9015 | 0.9101 | 0.9350 | 0.9015 |
| SimpleCNN | 0.8957 | 0.9042 | 0.9130 | 0.9399 | 0.9042 |

**Selected model: SimpleCNN.** Marginally outperforms ResNet-50 on all IAM metrics. ResNet-50's systematic DOUBLE_LINE over-prediction (~9.2% per class) is an ImageNet pretraining artifact. SimpleCNN is also lighter (7.07M vs ~25M params) and consistent with the binary task choice. Custom-50 generalisation evaluation pending.

---

## Training Configuration

| Setting | Binary (SimpleCNN) | Multiclass SimpleCNN | Multiclass ResNet-50 |
|---|---|---|---|
| Optimizer | Adam | Adam | Adam |
| Learning rate | 1e-3 | 1e-4 | 4e-4 |
| LR scheduler | ReduceLROnPlateau | ReduceLROnPlateau | ReduceLROnPlateau |
| Batch size | 64 | 64 | 256 |
| Max epochs | 100 | 100 | 100 |
| Early stop patience | 15 | 15 | 15 |
| Loss function | Weighted CE (1:8) | Cross-Entropy | Cross-Entropy |

---

## Custom-50 Evaluation

55 real word images photographed from paper with hand-drawn cross-outs.  
7 images per style (SCRATCH has 6). Ground truth: folder name.  
No train/val/test split — evaluation only.  
Same val_transform applied: PadToSquare → 224×224 → normalize.

The model never trained on these images. Performance here reflects real-world generalisation beyond the synthetic IAM distribution.

---

## Observations & Findings

### ZIG_ZAG Over-prediction (ResNet-50 Multiclass)
~9–10% of samples from every class are misclassified as ZIG_ZAG. WAVE is the worst case (12.9%). Angular strokes in other cross-out types visually resemble zig-zag patterns. Visible in the confusion matrix as a systematic off-diagonal column.

---

## Notebook Guide

| Notebook | What it does |
|---|---|
| `binary/01_simplecnn_binary.ipynb` | Train SimpleCNN for binary classification |
| `binary/02_binary_resnet.ipynb` | Train ResNet-50 for binary (exploratory) |
| `binary/02_binary_vit.ipynb` | Train ViT-B/16 for binary (exploratory) |
| `binary/02_binary_efficientnet.ipynb` | Train EfficientNet-B0 for binary (exploratory) |
| `multiclass/01_simplecnn_multiclass.ipynb` | Train SimpleCNN for 8-class classification |
| `multiclass/02_multiclass_resnet.ipynb` | Train ResNet-50 for 8-class classification |
| `eval/04_eval_iam_test.ipynb` | Evaluate all checkpoints on IAM test set |
| `checkpoints/04_inspect_checkpoints.ipynb` | Inspect saved checkpoint integrity and metrics |
| `crop_custom50.ipynb` | Prepare and crop the Custom-50 evaluation images |

---

## Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Resize strategy | PadToSquare → Resize(224) | Preserves stroke geometry; stretching distorts frequency/amplitude |
| Pad fill | White (255) | Matches IAM white background |
| Colour mode | Grayscale → 3-channel | Required for ImageNet pre-trained models |
| MIXED in binary | Yes (label=1) | MIXED is crossed-out, just not a specific style |
| MIXED in multiclass | Excluded | No clean label to assign |
| Class weighting | 8× for CLEAN | Compensates 1:8 imbalance in binary task |
| layer4 unfrozen (ResNet) | Yes (multiclass only) | Needs high-level feature adaptation; binary task too simple to need it |
| SimpleCNN no pool in blocks 3–4 | Deliberate | Preserves spatial detail for fine-grained stroke patterns |
| No RandomHorizontalFlip | Deliberate | Flipping mirrors the geometry of directional cross-outs (e.g. diagonal) |

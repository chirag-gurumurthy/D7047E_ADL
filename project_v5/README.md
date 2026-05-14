# D7047E ADL — Group 14: Cross-out Detection in Handwritten Documents

## Team
Suresh Balaraman, Nagarajan Ganesan, Chirag Gurumurthy, Shameena Mohammed Nabeel, Lakshmi Shankar Paramasivan

## Repository
https://github.com/chirag-gurumurthy/D7047E_ADL/tree/project

---

## Project Tasks

### Task 1: Binary Classification — Clean vs. Crossed-out
- Models: SimpleCNN (baseline), ResNet-50, EfficientNet-B0, ViT-B/16
- Dataset: CLEAN (label 0) vs ALL 7 styles + MIXED (label 1)
- Class imbalance handled via loss weighting

### Task 2: Multi-class Classification — 7 Cross-out Styles
- Models: SimpleCNN (baseline), SimpleCNNv2 (improved), ResNet-50, EfficientNet-B0, ViT-B/16
- 8 classes: CLEAN + 7 styles (MIXED excluded)

### Task 3: Custom-50 Evaluation
- ~50 hand-drawn cross-out images
- Evaluate generalization beyond synthetic training distribution

---

## Progress Tracker

| Task | Status | Notes |
|---|---|---|
| Data download & extraction | Done | 47,997 images/class, pre-split |
| Data preprocessing | Done | PadToSquare + grayscale→3ch + ImageNet norm |
| Task 1: Binary — SimpleCNN (baseline) | Done | best_binary_SimpleCNN.pth (epoch 10) |
| Task 1: Binary — ResNet-50 | Done | best_binary_ResNet_50.pth (epoch 10) |
| Task 1: Binary — EfficientNet-B0 | In Progress | Retraining with LR=1e-4 (overfits at 1e-3) |
| Task 1: Binary — ViT-B/16 | Done | best_binary_ViT_B_16.pth (epoch 10) |
| Task 1: Model comparison | TODO | |
| Task 2: Multi-class — SimpleCNN (baseline) | Done | best_mc_SimpleCNN.pth (epoch 30), F1=0.86 |
| Task 2: Multi-class — SimpleCNNv2 (improved) | TODO | Fixes Wave/ZigZag/SingleLine confusion |
| Task 2: Multi-class — ResNet-50 | Done | best_mc_ResNet_50.pth (epoch 25) |
| Task 2: Multi-class — EfficientNet-B0 | TODO | |
| Task 2: Multi-class — ViT-B/16 | Done | best_mc_ViT_B_16.pth (epoch 17) |
| Task 2: Model comparison | TODO | |
| Task 3: Custom-50 dataset creation | Done | See custom_50/ |
| Task 3: Custom-50 evaluation | Done | 03_eval_custom50.ipynb |

---

## Checkpoint Summary

| File | Task | Best Epoch | Val Loss |
|---|---|---|---|
| best_binary_SimpleCNN.pth | binary | 10 | 0.0797 |
| best_binary_ResNet_50.pth | binary | 10 | 0.0912 |
| best_binary_EfficientNet_B0.pth | binary | 1 | 0.1347 — retraining |
| best_binary_ViT_B_16.pth | binary | 10 | 0.0836 |
| best_mc_SimpleCNN.pth | multiclass | 30 | 0.3231 |
| best_mc_ResNet_50.pth | multiclass | 25 | 0.7162 |
| best_mc_ViT_B_16.pth | multiclass | 17 | 0.5759 |

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_simplecnn_baseline.ipynb` | SimpleCNN binary baseline |
| `01_simplecnn_multiclass.ipynb` | SimpleCNN multiclass baseline |
| `01_simplecnn_multiclass_v2.ipynb` | SimpleCNNv2 multiclass (improved architecture) |
| `02_binary.ipynb` | Binary: ResNet-50, EfficientNet-B0, ViT-B/16 |
| `02_multiclass_v3.ipynb` | Multiclass: ResNet-50, EfficientNet-B0, ViT-B/16 |
| `03_eval_custom50.ipynb` | Custom-50 evaluation |
| `04_inspect_checkpoints.ipynb` | Inspect all saved checkpoints (integrity + metrics + class balance) |

---

## Architecture Summary

### SimpleCNN (v1 — baseline)
- 3 conv layers, 3 MaxPool → feature map 28×28
- No BatchNorm
- Classifier: Linear(100352→256→8)

### SimpleCNNv2 (improved)
- 8 conv layers, 2 MaxPool + AdaptiveAvgPool → feature map 7×7
- BatchNorm after every conv
- Preserves spatial detail for Wave/ZigZag/SingleLine distinction
- Classifier: Linear(12544→512→128→8)

### Pretrained Models (ResNet-50, EfficientNet-B0, ViT-B/16)
- Frozen backbone + fine-tuned head
- LR: 1e-3 (1e-4 for EfficientNet-B0)
- Resume supported via `resume_path` in `train_model()`

---

## TODO: Update in Project Report/Document

- [ ] **Section 4.3**: Add final hyperparameters (LR, batch size, epochs, optimizer)
- [ ] **Section 4.3**: Document class imbalance handling (loss weighting ratio)
- [ ] **Section 4.3**: Add actual architecture comparison results (Table: Accuracy/F1 per model per task)
- [ ] **Section 3**: Add citation for EfficientNet (Tan & Le, 2019)
- [ ] **Section 4.1**: Confirm exact train/val/test counts after MIXED inclusion decision
- [ ] **Section 4.3**: Add training curves (loss/accuracy plots) as figures
- [ ] **Section 4.3**: Add confusion matrices for best model per task
- [ ] **Section 5**: Update with custom-50 dataset description and results
- [ ] Add discussion on Wave/ZigZag/SingleLine confusion and SimpleCNNv2 fix
- [ ] Add generalization analysis — synthetic vs real (custom-50) performance gap

---

## Observations & Findings

### Domain Shift — SimpleCNN Multiclass (2026-05-14)
- IAM test Macro-F1: **0.90** vs Custom-50 Macro-F1: **0.46** → gap of ~0.44
- Likely cause: IAM crossouts are clean digital-style scans; custom_50 are real photos with variable lighting, perspective, blur, and shadows
- Model learned features specific to the IAM domain and does not generalise to real-world capture conditions
- Potential fixes (not yet tried):
  - Stronger training augmentation: `RandomPerspective`, `GaussianBlur`, `ColorJitter`
  - Few-shot fine-tuning on a small number of custom_50 samples
  - Adding custom_50 images to the training mix

---

## Design Decisions Log

| Decision | Choice | Reason |
|---|---|---|
| Resize strategy | PadToSquare → Resize(224) | Preserves cross-out geometry |
| Pad fill | White (255) | Matches IAM background |
| Image inversion | No | Black ink on white — standard for pretrained CNNs |
| Colour mode | Grayscale → 3-channel | Required for ImageNet pretrained models |
| MIXED in training | Binary: Yes / Multi-class: No | MIXED is crossed-out but not a specific style |
| Word length (custom-50) | 4–7 letters | Matches IAM distribution; long enough for pattern visibility |
| EfficientNet-B0 LR | 1e-4 (not 1e-3) | Overfits at 1e-3 — val acc decreasing from epoch 1 |
| SimpleCNNv2 | 2 MaxPool instead of 3 | Preserves spatial detail for Wave/ZigZag distinction |

---

## Dataset Notes

- Images: ~27×51px grayscale, black ink on white
- Files are **paired** — same filename exists across all categories
- Splits: train (47,997/class), val (7,559/class), test (20,306/class)
- Dataset is **perfectly balanced** — equal samples per class in every split
- MIXED = randomly applied cross-out type per image (not a classification label)

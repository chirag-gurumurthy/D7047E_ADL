D7047E ADL — Group 14: Cross-out Detection in Handwritten Documents
====================================================================

TEAM
  Suresh Balaraman, Nagarajan Ganesan, Chirag Gurumurthy,
  Shameena Mohammed Nabeel, Lakshmi Shankar Paramasivan


=============================================================================
PART 1 — WHAT THE PROJECT DOES
=============================================================================

OVERVIEW
--------
We train neural networks to detect and classify cross-out marks on
handwritten word images. Two classification tasks:

  Task 1 — Binary:      Is the word crossed out or not? (2 classes)
  Task 2 — Multi-class: Which cross-out style is it?    (8 classes)
  Task 3 — Custom-50:   How well does the model work on real photos?


DATASET (IAM Cross-out)
-----------------------
Source: IAM Handwriting Database with synthetic cross-outs added on top.

~48,000 word images per class, pre-split into train / val / test.
Image size: ~27x51 px, grayscale, black ink on white background.

8 categories:
  CLEAN        no cross-out (baseline)
  SINGLE_LINE  one horizontal line through the word
  DOUBLE_LINE  two parallel horizontal lines
  DIAGONAL     one diagonal line
  CROSS        two crossing lines forming an X
  WAVE         a wavy horizontal line
  ZIG_ZAG      zig-zag line
  SCRATCH      heavy scribbling covering the word

+ MIXED: randomly chosen style per image — used only in binary Task 1
         (it is crossed-out, but not a classifiable style for Task 2)

The dataset is PAIRED: the exact same word image appears in every category
folder. e.g., "a01-000u-00-00.png" in CLEAN/ and in SINGLE_LINE/ is the
same word, with the cross-out drawn on top. This means every style was
applied to the same set of words — a controlled experiment.


=============================================================================
PART 2 — PREPROCESSING PIPELINE
=============================================================================

Every image goes through the same pipeline before the model sees it:

  Step 1 — Grayscale to 3-channel
    The images are grayscale (1 channel). Pretrained models (ResNet, etc.)
    expect 3 channels (RGB). We duplicate the grayscale channel 3 times.
    No colour information is added — it is still effectively grayscale.

  Step 2 — PadToSquare (fill=255 white)
    Word images are wide and short (~382x73 px). If we resize directly to
    224x224 the word gets stretched and distorted, which would also distort
    the cross-out geometry (a diagonal would no longer look diagonal).
    Instead we pad the shorter side symmetrically with white pixels to make
    it square first, then resize.

  Step 3 — Resize to 224x224
    Required input size for ResNet-50, EfficientNet-B0 and ViT-B/16.

  Step 4 — ToTensor
    Converts PIL image (0-255 integers) to a PyTorch float tensor (0.0-1.0).

  Step 5 — Normalize with ImageNet statistics
    Mean: [0.485, 0.456, 0.406]
    Std:  [0.229, 0.224, 0.225]
    Pretrained models were trained with this normalization. Using the same
    normalization keeps the input distribution consistent with what the
    pretrained weights expect.

Training only (augmentation to improve generalization):
  RandomRotation(±5°)            — simulates slightly tilted paper or camera angle
  RandomAffine(translate=5%)     — slight position shift for positional robustness
  Note: RandomHorizontalFlip is NOT used — flipping changes the visual meaning
  of cross-out styles (e.g. a diagonal would mirror direction).
Val / test / custom-50: no augmentation — deterministic transforms only.


=============================================================================
PART 3 — NETWORK ARCHITECTURES
=============================================================================

------------------------------------------------------------------------
MODEL 1: SimpleCNN (trained from scratch — baseline)
------------------------------------------------------------------------

Input: 3 x 224 x 224

Feature extraction (self.features):
  Conv2d(3→32,  3x3, padding=1) → ReLU → MaxPool2d(2)   224→112
  Conv2d(32→64, 3x3, padding=1) → ReLU → MaxPool2d(2)   112→56
  Conv2d(64→128,3x3, padding=1) → ReLU → MaxPool2d(2)    56→28

  Output: 128 feature maps of size 28x28 = 100,352 values

Classification head (self.classifier):
  Flatten → Linear(100352→256) → ReLU → Dropout(0.3) → Linear(256→2 or 8)

Why this is the baseline:
  No pretrained weights. Learns everything from the IAM data alone.
  Gives a lower-bound on performance — if transfer learning doesn't beat
  this, something is wrong.

What Conv2d does:
  A filter (small grid of learned numbers) slides across the image and
  computes a dot product at each position. This detects local patterns:
  edges, lines, curves. With 32 filters in layer 1, it learns 32 different
  pattern detectors simultaneously.

What MaxPool2d(2) does:
  Takes the maximum value in each 2x2 region, halving the spatial size.
  Makes the features slightly position-invariant (small shifts in the word
  don't change the result much) and reduces computation.

What ReLU does:
  Sets all negative values to 0. Introduces non-linearity — without it,
  stacking layers would be equivalent to a single linear transformation.

What Dropout(0.3) does:
  During training, randomly sets 30% of activations to 0. Forces the network
  to not rely on any single neuron — reduces overfitting.


------------------------------------------------------------------------
MODEL 2: ResNet-50 (transfer learning)
------------------------------------------------------------------------

Base model: ResNet-50 pretrained on ImageNet (1.2M images, 1000 classes)
Parameters: ~25.6M total

Architecture concept — residual connections:
  Standard CNNs: x → Conv → Conv → output
  ResNet:        x → Conv → Conv → output + x   (skip connection)

  The "+ x" (adding the input directly to the output) solves the vanishing
  gradient problem in very deep networks. Gradients can flow back through
  the skip connection directly without passing through all the conv layers.
  This allows training networks with 50, 100, even 150+ layers.

How we use it (frozen backbone + new head):
  1. Load weights pretrained on ImageNet
  2. Freeze all backbone layers (param.requires_grad = False)
     → These weights do NOT change during training
  3. Replace the final fully-connected layer with a new head:

     Binary:     Linear(2048→256) → ReLU → Dropout(0.3) → Linear(256→2)
     Multi-class: Linear(2048→512) → ReLU → Dropout(0.3) → Linear(512→8)

  Only the new head trains (~0.5M parameters out of 25.6M).

Why freeze the backbone?
  The pretrained backbone already knows how to detect edges, textures,
  shapes — general visual features. Cross-out detection needs exactly these
  features. Training the full network would require much more data and time,
  and risks "forgetting" the pretrained features (catastrophic forgetting).


------------------------------------------------------------------------
MODEL 3: EfficientNet-B0 (transfer learning)
------------------------------------------------------------------------

Base model: EfficientNet-B0 pretrained on ImageNet
Parameters: ~5.3M total (much smaller than ResNet-50)

Architecture concept — compound scaling:
  Traditional scaling: make network deeper OR wider OR use higher resolution.
  EfficientNet: scale all three dimensions simultaneously using a fixed ratio.
  Result: better accuracy per parameter than ResNet.

How we use it:
  1. Load pretrained weights, freeze backbone
  2. Replace classifier head:

     Binary:      Dropout(0.3) → Linear(1280→2)
     Multi-class: Dropout(0.3) → Linear(1280→8)

  The input to the head is 1280 features (EfficientNet-B0's output size).


------------------------------------------------------------------------
MODEL 4: ViT-B/16 (Vision Transformer — transfer learning)
------------------------------------------------------------------------

Base model: ViT-B/16 pretrained on ImageNet
Parameters: ~86M total

Architecture concept — no convolutions at all:
  Step 1 — Patch embedding:
    Split the 224x224 image into a grid of 16x16 patches → 14x14 = 196 patches
    Each patch (16x16x3 = 768 values) is projected to a 768-d vector.
    This gives a sequence of 196 vectors — like words in a sentence.

  Step 2 — Add positional encoding:
    The Transformer doesn't know the order/position of patches by default.
    A learned position embedding is added to each patch vector.

  Step 3 — Transformer encoder (12 layers):
    Each layer has two parts:
      a) Multi-head Self-Attention: each patch "looks at" all other patches
         and weights them by relevance. This captures global relationships —
         a cross-out line at the top can influence the representation of
         patches at the bottom.
      b) Feed-forward network: applied independently to each patch.

  Step 4 — Classification token [CLS]:
    A special learnable token is prepended to the sequence. After 12
    transformer layers, this token's output is used for classification.

How we use it:
  1. Load pretrained weights, freeze backbone
  2. Replace the head:

     Binary:      Linear(768→2)
     Multi-class: Linear(768→8)

Why ViT might be interesting for this task:
  Cross-out marks can span the entire word. A CNN sees local patches first
  and builds up global context gradually. ViT sees all patches simultaneously
  from the first layer — potentially better at capturing long horizontal
  or diagonal lines that span the word.


=============================================================================
PART 4 — TRAINING SETUP
=============================================================================

Loss function:
  Task 1 (binary): Weighted CrossEntropyLoss
    CLEAN images are 1 class out of 9 crossed-out classes (including MIXED).
    The weight for CLEAN is set higher so misclassifying CLEAN is penalised
    more. This compensates for class imbalance.

  Task 2 (multi-class): Standard CrossEntropyLoss
    8 balanced classes — no weighting needed.

  CrossEntropyLoss measures how wrong the predicted probabilities are
  compared to the true label. It is the standard loss for classification.

Optimizer: Adam (lr=1e-3)
  Adapts the learning rate per parameter based on gradient history.
  More robust than plain SGD — good default for most classification tasks.

Learning rate scheduler: StepLR (step=10, gamma=0.1)
  Every 10 epochs, multiply the learning rate by 0.1.
  Epochs 1-10: lr=1e-3, Epochs 11-20: lr=1e-4, Epochs 21-30: lr=1e-5...
  Helps fine-tune in the later stages of training.

Early stopping:
  Min epochs: 50  — early stopping cannot trigger before epoch 50.
  Patience: 40    — stop after 40 consecutive epochs without val_loss improvement.
  Restore the best weights seen during training.
  Prevents overfitting and wastes no time training past the optimal point.

Print schedule: epoch 1, then every 10th epoch (10, 20, 30...).
  Save/early-stop events always printed.

Max epochs:  100
Min epochs:  50
Batch size:  64
Num workers: 8


=============================================================================
PART 5 — EVALUATION METRICS
=============================================================================

Accuracy
  Fraction of correct predictions. Simple but misleading when classes are
  imbalanced.

Precision (per class)
  Of all images predicted as class X, how many actually are X?
  High precision = few false positives.

Recall (per class)
  Of all images that actually are class X, how many did we correctly find?
  High recall = few false negatives.

F1 Score
  Harmonic mean of precision and recall: 2 * P * R / (P + R).
  Balanced measure — penalises both low precision and low recall.
  Use macro-F1 (average over classes equally) to compare models fairly.

Confusion Matrix
  NxN table where entry (i,j) = number of images from class i predicted
  as class j. Diagonal = correct predictions. Off-diagonal = errors.
  Shows WHICH classes get confused with each other.

Generalisation Gap (Task 3)
  IAM test F1 - Custom-50 F1.
  Measures how much the model relies on synthetic-data patterns that do not
  appear in real handwriting. A large gap = poor generalisation.


=============================================================================
PART 6 — TASK 3: CUSTOM-50
=============================================================================

55 real word images, photographed from paper with hand-drawn cross-outs.
7 images per style (SCRATCH has 6).

Ground truth: folder name (CLEAN/, SINGLE_LINE/, etc.)
No train/val/test split — evaluation only.
Same val_transform applied: PadToSquare → 224x224 → normalize.

The model never trained on these images. Performance here reflects real-world
generalisation beyond the synthetic IAM distribution.

Files: custom_50/CLEAN/, custom_50/SINGLE_LINE/, ... custom_50/SCRATCH/


=============================================================================
PART 7 — CONCEPTS TO STUDY
=============================================================================

If you want to understand this project more deeply, these are the key topics:

1. Backpropagation & Gradient Descent
   How neural networks learn: compute loss, calculate gradients of loss
   w.r.t. every weight, update weights in the direction that reduces loss.
   Resource: "Neural Networks and Deep Learning" ch.2 (Nielsen, free online)

2. Convolutional Neural Networks
   How Conv2d, MaxPool, and feature maps work visually.
   Resource: CS231n Stanford — "Convolutional Neural Networks" lecture notes

3. Transfer Learning & Fine-tuning
   Why pretrained features transfer, when to freeze vs unfreeze.
   Resource: fast.ai Practical Deep Learning — Lesson 1-2

4. Residual Networks (ResNet)
   Why skip connections solve vanishing gradients in deep networks.
   Paper: He et al. 2015 "Deep Residual Learning for Image Recognition"

5. Attention Mechanism & Transformers
   How self-attention computes which parts of the input to focus on.
   Resource: "The Illustrated Transformer" (Jay Alammar, blog post)

6. Vision Transformers (ViT)
   How the Transformer architecture is adapted for images using patches.
   Paper: Dosovitskiy et al. 2020 "An Image is Worth 16x16 Words"

7. EfficientNet
   Compound scaling of width/depth/resolution.
   Paper: Tan & Le 2019 "EfficientNet: Rethinking Model Scaling for CNNs"

8. Class Imbalance
   Why accuracy is misleading and how weighted loss helps.
   Understand: precision, recall, F1, confusion matrix.


=============================================================================
FILES
=============================================================================

Shared module:
  common.py                      transforms, datasets, train_model() — shared by all notebooks

Phase 1 — SimpleCNN baseline (run both in parallel):
  01_simplecnn_binary.ipynb      SimpleCNN, binary task
  01_simplecnn_multiclass.ipynb  SimpleCNN, multiclass task

Phase 2 — Pretrained models (run both in parallel, after Phase 1):
  02_binary.ipynb                ResNet-50, EfficientNet-B0, ViT-B/16 — binary task
  02_multiclass.ipynb            ResNet-50, EfficientNet-B0, ViT-B/16 — multiclass task

Phase 3 — Evaluation (after Phase 2):
  03_eval_custom50.ipynb         loads best multiclass checkpoint → evaluates on custom-50

Cluster:
  run.sh                         runs all notebooks in the correct order with parallelism

Data / assets:
  common.py                      shared preprocessing, dataset classes, training loop
  checkpoints/                   saved model checkpoints (.pth) — created at runtime
  iam_crossouts/                 IAM dataset (train/val/test) — downloaded at runtime
  custom_50/                     custom-50 real-photo dataset (8 subfolders, 55 images)
  raw_photos/                    original photos (crossout1.png, crossout2.png)
  crop_custom50.ipynb            crops raw photos into custom_50/ subfolders
  README.txt                     this file

WandB experiment tracking:
  Project: adl-crossouts-v3
  Groups:  binary / multiclass / custom
  Names:   SimpleCNN / ResNet-50 / EfficientNet-B0 / ViT-B/16

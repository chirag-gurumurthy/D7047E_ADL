"""
common.py — Shared code for all training/evaluation notebooks.

             02_binary.ipynb, 02_multiclass.ipynb, 03_eval_custom50.ipynb
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import wandb

# ── Constants ──────────────────────────────────────────────────────────────────

WANDB_PROJECT = 'adl-crossout-v5'
WANDB_GROUP_BINARY     = 'binary'
WANDB_GROUP_MULTICLASS = 'multiclass'

CATEGORIES = [
    'CLEAN', 'SINGLE_LINE', 'DOUBLE_LINE', 'DIAGONAL',
    'CROSS', 'WAVE', 'ZIG_ZAG', 'SCRATCH',
]
BINARY_CROSSED = [
    'SINGLE_LINE', 'DOUBLE_LINE', 'DIAGONAL', 'CROSS',
    'WAVE', 'ZIG_ZAG', 'SCRATCH', 'MIXED',
]


# ── Transforms ─────────────────────────────────────────────────────────────────

class PadToSquare:
    """Symmetrically pad image to square with white fill, preserving aspect ratio."""
    def __init__(self, fill=255):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        s = max(w, h)
        pl, pt = (s - w) // 2, (s - h) // 2
        return TF.pad(img, (pl, pt, s - w - pl, s - h - pt), fill=self.fill)


def get_transforms(img_size=224):
    """
    Returns (train_transform, val_transform).

    Train augmentation:
      RandomRotation(5°)            — simulates slightly angled paper/camera
      RandomAffine(5% shift)        — slight position shift for robustness
      RandomPerspective(0.2, p=0.5) — simulates angled photo capture
      GaussianBlur(sigma 0.1–1.0)   — simulates camera blur / low resolution
      ColorJitter(brightness/contrast 0.3) — simulates lighting variation
    Val: deterministic only.
    """
    train_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        PadToSquare(fill=255),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=255),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=255),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_t = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        PadToSquare(fill=255),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_t, val_t


# ── Datasets ───────────────────────────────────────────────────────────────────

class CrossOutDataset(Dataset):
    """8-class multiclass dataset (CLEAN + 7 styles). MIXED excluded."""
    def __init__(self, base_path, categories, transform=None):
        self.transform = transform
        self.samples   = []
        label_map = {cat: i for i, cat in enumerate(categories)}
        for cat in categories:
            folder = os.path.join(base_path, cat)
            if not os.path.exists(folder):
                print(f'Warning: {folder} not found')
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder, fname), label_map[cat]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img) if self.transform else img, label


class BinaryDataset(Dataset):
    """Binary dataset: CLEAN=0, all crossed-out styles + MIXED=1."""
    def __init__(self, base_path, transform=None):
        self.transform = transform
        self.samples   = []
        clean_dir = os.path.join(base_path, 'CLEAN')
        for fname in os.listdir(clean_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append((os.path.join(clean_dir, fname), 0))
        for cat in BINARY_CROSSED:
            folder = os.path.join(base_path, cat)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder, fname), 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        return self.transform(img) if self.transform else img, label


# ── Models ─────────────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Improved SimpleCNN for crossout classification.

    Changes vs v1:
    - BatchNorm after every conv (stable training, reduces early bias)
    - 4 conv blocks instead of 3 (more capacity)
    - Only 2 MaxPool layers instead of 3 (preserves spatial detail for
      fine-grained patterns like Wave vs ZigZag vs Single Line)
    - Deeper classifier head (512 hidden units)
    - Trained from scratch, no pretrained weights
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1 — 224×224 → 112×112
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2 — 112×112 → 56×56
            nn.Conv2d(32,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3 — 56×56 (no pool — preserve spatial detail)
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),

            # Block 4 — 56×56 (no pool — preserve spatial detail)
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),  # 56×56×256 → 7×7×256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),             # 7×7×256 = 12,544
            nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training loop ──────────────────────────────────────────────────────────────

def train_model(model_name, model, train_loader, val_loader, criterion,
                task, save_path, device,
                lr, epochs, min_epochs, patience,
                wandb_project, wandb_group, batch_size,
                lr_patience=5, lr_factor=0.1, min_lr=1e-6,
                resume_path=None):
    """
    Train a model with early stopping.

    Early stopping only triggers after `min_epochs`.
    Best checkpoint (lowest val_loss) is saved to `save_path`.

    WandB logging: project=wandb_project, group=wandb_group, name=model_name

    resume_path: path to a .pth checkpoint to resume from. Loads model weights,
                 optimizer state, scheduler state, and continues from saved epoch.
    """
    model = model.to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )

    start_epoch = 1
    best_val_loss = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                print('  Warning: scheduler state incompatible, starting fresh scheduler.')
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        no_improve    = ckpt.get('no_improve', 0)
        history['train_loss'] = ckpt.get('train_loss', [])
        history['train_acc']  = ckpt.get('train_acc',  [])
        history['val_loss']   = ckpt.get('val_loss_hist', [])
        history['val_acc']    = ckpt.get('val_acc',    [])
        print(f'  Resumed from {resume_path}')
        print(f'  Starting at epoch {start_epoch} | best val_loss so far: {best_val_loss:.6f}')

    run = wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=model_name,
        config=dict(
            model=model_name, task=task, lr=lr, batch_size=batch_size,
            epochs=epochs, min_epochs=min_epochs, patience=patience,
            lr_patience=lr_patience, lr_factor=lr_factor, min_lr=min_lr,
        ),
        reinit=True,
    )

    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f'\n=== {model_name} [{task}] === '
          f'(max {epochs} ep | min {min_epochs} | patience {patience})')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tl += loss.item() * imgs.size(0)
            tc += (out.argmax(1) == labels).sum().item()
            tt += imgs.size(0)

        model.eval()
        vl, vc, vt = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                vl += loss.item() * imgs.size(0)
                vc += (out.argmax(1) == labels).sum().item()
                vt += imgs.size(0)

        tr_loss, tr_acc = tl / tt, tc / tt
        vl_loss, vl_acc = vl / vt, vc / vt
        scheduler.step(vl_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        wandb.log({
            'train_loss': tr_loss, 'train_acc': tr_acc,
            'val_loss': vl_loss,   'val_acc': vl_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch,
        })

        saved = False
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            torch.save({
                'model_state_dict':     best_weights,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_name':           model_name,
                'task':                 task,
                'categories':           CATEGORIES,
                'epoch':                epoch,
                'val_loss':             best_val_loss,
                'no_improve':           0,
                'train_loss':           history['train_loss'],
                'train_acc':            history['train_acc'],
                'val_loss_hist':        history['val_loss'],
                'val_acc':              history['val_acc'],
            }, save_path)
            saved = True
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or saved:
            tag = ' [saved]' if saved else f' (no improve {no_improve}/{patience})'
            print(f'  Ep {epoch:>3} | '
                  f'Train {tr_loss:.4f}/{tr_acc:.3f} | '
                  f'Val {vl_loss:.4f}/{vl_acc:.3f}{tag}')

        if epoch >= min_epochs and no_improve >= patience:
            print(f'  Early stop at epoch {epoch}')
            break

    model.load_state_dict(best_weights)
    wandb.summary['best_val_loss'] = best_val_loss
    run.finish()
    print(f'  Done. val_loss={best_val_loss:.4f} | '
          f'{time.time() - t0:.0f}s | {save_path}')
    return model, history


def log_test_metrics(preds, labels, probs=None, model_name=None,
                     wandb_project=None, wandb_group=None, task='multiclass'):
    """Log final test metrics to WandB summary."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, roc_auc_score)
    avg = 'binary' if task == 'binary' else 'macro'
    metrics = {
        'test_accuracy':  accuracy_score(labels, preds),
        'test_precision': precision_score(labels, preds, average=avg, zero_division=0),
        'test_recall':    recall_score(labels, preds, average=avg, zero_division=0),
        'test_f1':        f1_score(labels, preds, average=avg, zero_division=0),
    }
    if probs is not None and task == 'binary':
        metrics['test_auc'] = roc_auc_score(labels, probs)

    run = wandb.init(
        project=wandb_project, group=wandb_group,
        name=f'{model_name}-test-metrics', reinit=True,
    )
    wandb.summary.update(metrics)
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')
    run.finish()
    return metrics


def rebuild_model(model_name, nc):
    """Rebuild multiclass model architecture for loading checkpoints."""
    from torchvision import models
    if model_name == 'SimpleCNN':
        return SimpleCNN(nc)
    if model_name == 'ResNet-50':
        m = models.resnet50(weights=None)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,nc))
        return m
    if model_name == 'ViT-B/16':
        m = models.vit_b_16(weights=None)
        m.heads = nn.Linear(768, nc)
        return m
    raise ValueError(f'Unknown model: {model_name}')


def rebuild_binary_model(model_name):
    """Rebuild binary model architecture for loading checkpoints."""
    from torchvision import models
    if model_name == 'SimpleCNN':
        return SimpleCNN(2)
    if model_name == 'ResNet-50':
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Linear(m.fc.in_features,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,2))
        return m
    if model_name == 'ViT-B/16':
        m = models.vit_b_16(weights=None)
        m.heads = nn.Linear(768, 2)
        return m
    raise ValueError(f'Unknown model: {model_name}')


def log_confusion_matrix(preds, labels, class_names, model_name, wandb_project, wandb_group):
    """Log confusion matrix to WandB as a matplotlib heatmap image."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), max(5, len(class_names))))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()

    run = wandb.init(
        project=wandb_project, group=wandb_group,
        name=f'{model_name}-confusion-matrix', reinit=True,
    )
    wandb.log({'confusion_matrix': wandb.Image(fig)})
    plt.close(fig)
    run.finish()

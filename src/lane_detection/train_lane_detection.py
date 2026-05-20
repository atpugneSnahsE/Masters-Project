"""
TuSimple Lane Detection v2 — U-Net (ResNet34 encoder)
Improvements over v1:
  - ResNet34 encoder (more powerful)
  - 50 epochs
  - Weighted Dice+BCE loss (handles lane pixel class imbalance)
  - Starts fresh with ResNet34 (different architecture from v1)
  - Lower initial LR with warmup via cosine schedule
  - Larger lane mask thickness for better signal

Usage:
    python3 train_lane_detection_v2.py
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT  = os.path.expanduser("~/Downloads/archive/TUSimple/train_set")
LABEL_FILES   = [
    "label_data_0313.json",
    "label_data_0531.json",
    "label_data_0601.json",
]
IMG_H, IMG_W  = 368, 640
LANE_WIDTH    = 14             # thicker than v1 (was 10) — more signal for sparse lanes
BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LR            = 5e-4           # lower than v1 (was 1e-3) — more stable with resnet34
VAL_SPLIT     = 0.1
SAVE_DIR      = os.path.expanduser("~/lane_model_v2")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Loss weights — increase Dice weight to combat class imbalance
BCE_WEIGHT    = 1.0
DICE_WEIGHT   = 3.0            # was effectively 1.0 in v1

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Using device : {DEVICE}")
print(f"Encoder      : resnet34")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Lane width   : {LANE_WIDTH}px")
print(f"Loss weights : BCE={BCE_WEIGHT}  Dice={DICE_WEIGHT}")
print("-" * 55)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class TuSimpleDataset(Dataset):
    def __init__(self, records, dataset_root, transform=None):
        self.records      = records
        self.dataset_root = dataset_root
        self.transform    = transform

    def __len__(self):
        return len(self.records)

    def _make_mask(self, lanes, h_samples, img_h, img_w):
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for lane in lanes:
            points = []
            for x, y in zip(lane, h_samples):
                if x == -2:
                    continue
                points.append((int(x), int(y)))
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    cv2.line(mask, points[i], points[i+1],
                             color=1, thickness=LANE_WIDTH)
            elif len(points) == 1:
                cv2.circle(mask, points[0], LANE_WIDTH // 2,
                           color=1, thickness=-1)
        return mask

    def __getitem__(self, idx):
        record   = self.records[idx]
        img_path = os.path.join(self.dataset_root, record["raw_file"])
        img_bgr  = cv2.imread(img_path)

        if img_bgr is None:
            img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            mask = np.zeros((IMG_H, IMG_W),    dtype=np.uint8)
        else:
            img_h, img_w = img_bgr.shape[:2]
            img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = self._make_mask(record["lanes"], record["h_samples"],
                                   img_h, img_w)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented["image"]
            mask = augmented["mask"].float()

        return img, mask


# ─────────────────────────────────────────────
# AUGMENTATIONS  (more aggressive than v1)
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3,
                               contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=7, p=0.3),
    A.RandomShadow(p=0.3),              # simulate shadows on road
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# LOSS  —  weighted Dice + BCE
# ─────────────────────────────────────────────
class WeightedDiceBCELoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=3.0, smooth=1.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.smooth      = smooth
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(1, 2))
        dice_loss = 1 - (2 * inter + self.smooth) / \
                    (probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + self.smooth)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss.mean()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_iou(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - inter
    iou   = (inter + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
all_records = []
for lf in LABEL_FILES:
    path = os.path.join(DATASET_ROOT, lf)
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        continue
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

print(f"Total annotated frames : {len(all_records)}")

np.random.seed(42)
np.random.shuffle(all_records)
split      = int(len(all_records) * (1 - VAL_SPLIT))
train_recs = all_records[:split]
val_recs   = all_records[split:]
print(f"Train : {len(train_recs)}  |  Val : {len(val_recs)}")
print("-" * 55)

train_ds = TuSimpleDataset(train_recs, DATASET_ROOT, transform=train_transform)
val_ds   = TuSimpleDataset(val_recs,   DATASET_ROOT, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)


# ─────────────────────────────────────────────
# MODEL  —  U-Net with ResNet34 encoder
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = 1,
)
model = model.to(DEVICE)
print(f"Model loaded: U-Net (ResNet34) on {DEVICE}")
print("-" * 55)

criterion = WeightedDiceBCELoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history  = {"train_loss": [], "val_loss": [], "val_iou": []}
best_iou = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # ── Validate ──
    model.eval()
    val_loss = 0.0
    val_iou  = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, masks)
            val_loss += loss.item() * imgs.size(0)
            val_iou  += compute_iou(logits, masks) * imgs.size(0)
    val_loss /= len(val_ds)
    val_iou  /= len(val_ds)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    marker = "  ✓ best" if val_iou > best_iou else ""
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
          f"train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  "
          f"val_iou={val_iou:.4f}{marker}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))

# Save final
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
print("-" * 55)
print(f"Training complete.  Best val IoU : {best_iou:.4f}")
print(f"Models saved to    : {SAVE_DIR}")


# ─────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, history["val_iou"], label="Val IoU",
         marker="o", markersize=3, color="green")
ax2.axhline(y=0.75, color="red", linestyle="--", label="Target IoU (0.75)")
ax2.axhline(y=best_iou, color="blue", linestyle=":",
            label=f"Best IoU ({best_iou:.4f})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("IoU")
ax2.set_title("Validation IoU")
ax2.legend()
ax2.grid(True)

plt.suptitle("U-Net (ResNet34) v2 — TuSimple Lane Detection", fontsize=13)
plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, "training_curves_v2.png")
plt.savefig(curve_path, dpi=150)
print(f"Training curves saved to : {curve_path}")
plt.show()
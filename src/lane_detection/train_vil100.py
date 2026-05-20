"""
VIL-100 Lane Detection — U-Net (ResNet34)
Optimised for MacBook Pro i7 CPU (no GPU)

Dataset structure expected:
  ~/Downloads/VIL100/
      JPEGImages/  <clip_name>/  00000.jpg ...
      Json/        <clip_name>/  00000.jpg.json ...
      Annotations/ <clip_name>/  00000.png ...

Usage:
    python3 train_vil100.py
"""

import os
import json
import glob
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
# CONFIG  —  tuned for MacBook i7 32GB RAM
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/VIL100")
SAVE_DIR     = os.path.expanduser("~/vil100_model")
IMG_H        = 368
IMG_W        = 640
LANE_WIDTH   = 10       # thickness when drawing lane mask from points
BATCH_SIZE   = 4        # keep low for CPU — increase to 6 if RAM allows
NUM_EPOCHS   = 30       # 30 epochs is enough with a pretrained encoder
LR           = 3e-4
VAL_SPLIT    = 0.1
BCE_WEIGHT   = 1.0
DICE_WEIGHT  = 3.0
DEVICE       = "cpu"    # MacBook i7 has no CUDA

# How many clips to use for training
# VIL100 has ~100 clips x 100 frames = ~10,000 images
# Set to None to use ALL clips (slower but better)
# Set to 50 to use half the dataset (good starting point)
MAX_CLIPS    = None

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 55)
print("  VIL-100 Lane Detection Training")
print("=" * 55)
print(f"  Device     : {DEVICE}")
print(f"  Image size : {IMG_W} x {IMG_H}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {NUM_EPOCHS}")
print(f"  Save dir   : {SAVE_DIR}")
print("=" * 55)


# ─────────────────────────────────────────────
# BUILD FILE LIST
# ─────────────────────────────────────────────
def build_file_list(dataset_root, max_clips=None):
    """
    Walk through Json/ folder and collect all (image_path, json_path) pairs.
    Returns list of dicts with keys: image_path, json_path
    """
    json_root  = os.path.join(dataset_root, "Json")
    img_root   = os.path.join(dataset_root, "JPEGImages")

    clip_dirs = sorted(os.listdir(json_root))
    if max_clips:
        clip_dirs = clip_dirs[:max_clips]

    records = []
    for clip in clip_dirs:
        clip_json_dir = os.path.join(json_root, clip)
        if not os.path.isdir(clip_json_dir):
            continue
        for jf in sorted(os.listdir(clip_json_dir)):
            if not jf.endswith(".json"):
                continue
            json_path = os.path.join(clip_json_dir, jf)
            # image filename: remove .json suffix → 00000.jpg
            img_name  = jf.replace(".json", "")   # 00000.jpg
            img_path  = os.path.join(img_root, clip, img_name)
            if os.path.exists(img_path):
                records.append({
                    "image_path": img_path,
                    "json_path" : json_path,
                })

    print(f"  Found {len(records)} annotated frames across {len(clip_dirs)} clips")
    return records


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class VIL100Dataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def _make_mask(self, json_path, img_h, img_w):
        """
        Read lane points from JSON and draw them as thick lines on a binary mask.
        Points are in original resolution (1920x1080) — scale to img_h x img_w.
        """
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        with open(json_path) as f:
            data = json.load(f)

        orig_w = data["info"]["width"]   # usually 1920
        orig_h = data["info"]["height"]  # usually 1080

        scale_x = img_w / orig_w
        scale_y = img_h / orig_h

        lanes = data["annotations"].get("lane", [])
        for lane in lanes:
            points = lane.get("points", [])
            if len(points) < 2:
                continue
            # Scale points to target resolution
            scaled = [(int(p[0] * scale_x), int(p[1] * scale_y))
                      for p in points]
            # Draw lane as thick polyline
            for i in range(len(scaled) - 1):
                cv2.line(mask, scaled[i], scaled[i+1],
                         color=1, thickness=LANE_WIDTH)

        return mask

    def __getitem__(self, idx):
        rec      = self.records[idx]
        img_bgr  = cv2.imread(rec["image_path"])

        if img_bgr is None:
            # Return blank if image missing
            img  = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            mask = np.zeros((IMG_H, IMG_W),    dtype=np.uint8)
        else:
            img_h, img_w = img_bgr.shape[:2]
            img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mask = self._make_mask(rec["json_path"], img_h, img_w)

        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img  = aug["image"]
            mask = aug["mask"].float()

        return img, mask


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3,
                               contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.2),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self, bce_w=1.0, dice_w=3.0, smooth=1.0):
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        inter    = (probs * targets).sum(dim=(1, 2))
        dice     = 1 - (2 * inter + self.smooth) / \
                   (probs.sum(dim=(1,2)) + targets.sum(dim=(1,2)) + self.smooth)
        return self.bce_w * bce_loss + self.dice_w * dice.mean()


# ─────────────────────────────────────────────
# METRIC
# ─────────────────────────────────────────────
def compute_iou(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1,2)) + targets.sum(dim=(1,2)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
all_records = build_file_list(DATASET_ROOT, max_clips=MAX_CLIPS)

if len(all_records) == 0:
    raise RuntimeError(
        "No records found! Check that DATASET_ROOT is correct: " + DATASET_ROOT)

np.random.seed(42)
np.random.shuffle(all_records)
split      = int(len(all_records) * (1 - VAL_SPLIT))
train_recs = all_records[:split]
val_recs   = all_records[split:]

print(f"  Train frames : {len(train_recs)}")
print(f"  Val frames   : {len(val_recs)}")
print("=" * 55) resnet

# num_workers=0 is important on macOS to avoid multiprocessing issues
train_ds     = VIL100Dataset(train_recs, transform=train_transform)
val_ds       = VIL100Dataset(val_recs,   transform=val_transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)


# ─────────────────────────────────────────────
# MODEL  —  freeze encoder to save time on CPU
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = "imagenet",   # pretrained — much better starting point
    in_channels     = 3,
    classes         = 1,
)

# Freeze encoder — only train the decoder (much faster on CPU)
for param in model.encoder.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable params : {trainable:,} / {total:,}  (encoder frozen)")
print("=" * 55)

model     = model.to(DEVICE)
criterion = DiceBCELoss(bce_w=BCE_WEIGHT, dice_w=DICE_WEIGHT)
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4)
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
    for batch_idx, (imgs, masks) in enumerate(train_loader):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch:02d}  batch {batch_idx+1}/{len(train_loader)}"
                  f"  loss={loss.item():.4f}", end="\r")

    train_loss /= len(train_ds)

    # ── Validate ──
    model.eval()
    val_loss = 0.0
    val_iou  = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits      = model(imgs).squeeze(1)
            val_loss   += criterion(logits, masks).item() * imgs.size(0)
            val_iou    += compute_iou(logits, masks) * imgs.size(0)
    val_loss /= len(val_ds)
    val_iou  /= len(val_ds)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    marker = "  ← best" if val_iou > best_iou else ""
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS}"
          f"  train={train_loss:.4f}"
          f"  val={val_loss:.4f}"
          f"  iou={val_iou:.4f}{marker}          ")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(),
                   os.path.join(SAVE_DIR, "best_model.pth"))

# Save final checkpoint
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
print("=" * 55)
print(f"  Training complete!")
print(f"  Best val IoU : {best_iou:.4f}")
print(f"  Model saved  : {SAVE_DIR}")
print("=" * 55)


# ─────────────────────────────────────────────
# PLOT TRAINING CURVES
# ─────────────────────────────────────────────
epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, history["val_iou"], color="green",
         label="Val IoU", marker="o", markersize=3)
ax2.axhline(y=best_iou, color="blue", linestyle=":",
            label=f"Best IoU ({best_iou:.4f})")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("IoU")
ax2.set_title("Validation IoU")
ax2.legend()
ax2.grid(True)

plt.suptitle("VIL-100 Lane Detection — U-Net ResNet34", fontsize=13)
plt.tight_layout()

curve_path = os.path.join(SAVE_DIR, "training_curves.png")
plt.savefig(curve_path, dpi=150)
print(f"  Training curves saved: {curve_path}")
plt.show()
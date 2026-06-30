"""
Lane Detection Diagnostics
Checks:
  1. Lane pixel ratio (class imbalance)
  2. What the model actually predicts visually
  3. Whether IoU is being calculated correctly for thin structures
  4. TuSimple official accuracy metric vs our IoU
"""

import os
import json
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/archive/TUSimple/train_set")
MODEL_PATH   = os.path.expanduser("~/lane_model_v2/best_model.pth")
LABEL_FILE   = os.path.join(DATASET_ROOT, "label_data_0313.json")
IMG_H, IMG_W = 368, 640
LANE_WIDTH   = 14
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def make_mask(lanes, h_samples, img_h, img_w, thickness=LANE_WIDTH):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for lane in lanes:
        points = [(int(x), int(y)) for x, y in zip(lane, h_samples) if x != -2]
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 1, thickness)
        elif len(points) == 1:
            cv2.circle(mask, points[0], thickness // 2, 1, -1)
    return mask

val_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = smp.Unet(
    encoder_name    = "resnet34",
    encoder_weights = None,
    in_channels     = 3,
    classes         = 1,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"Model loaded from: {MODEL_PATH}")

# ─────────────────────────────────────────────
# LOAD RECORDS
# ─────────────────────────────────────────────
records = []
with open(LABEL_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"Loaded {len(records)} records")

# ─────────────────────────────────────────────
# DIAGNOSTIC 1: Class imbalance
# ─────────────────────────────────────────────
print("\n── DIAGNOSTIC 1: Lane pixel ratio ──")
ratios = []
for rec in records[:200]:
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    mask = make_mask(rec["lanes"], rec["h_samples"], h, w)
    ratios.append(mask.sum() / (h * w))

ratios = np.array(ratios)
print(f"  Mean lane pixel ratio : {ratios.mean()*100:.2f}%")
print(f"  Min                   : {ratios.min()*100:.2f}%")
print(f"  Max                   : {ratios.max()*100:.2f}%")
print(f"  → Background:Lane ratio ≈ {(1-ratios.mean())/ratios.mean():.0f}:1")

# ─────────────────────────────────────────────
# DIAGNOSTIC 2: Threshold sensitivity
# ─────────────────────────────────────────────
print("\n── DIAGNOSTIC 2: IoU vs threshold sweep ──")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_ious = {t: [] for t in thresholds}

sample_records = records[:100]
for rec in sample_records:
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_orig = make_mask(rec["lanes"], rec["h_samples"], img_h, img_w)

    aug = val_transform(image=img_rgb, mask=mask_orig)
    img_t  = aug["image"].unsqueeze(0).to(DEVICE)
    mask_t = aug["mask"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(img_t).squeeze()
        prob  = torch.sigmoid(logit)

    for t in thresholds:
        pred  = (prob > t).float()
        inter = (pred * mask_t.squeeze()).sum()
        union = pred.sum() + mask_t.squeeze().sum() - inter
        iou   = (inter + 1e-6) / (union + 1e-6)
        thresh_ious[t].append(iou.item())

print(f"  {'Threshold':>10}  {'Mean IoU':>10}")
best_t, best_iou_t = 0.5, 0.0
for t in thresholds:
    mean_iou = np.mean(thresh_ious[t])
    marker = " ← best" if mean_iou > best_iou_t else ""
    if mean_iou > best_iou_t:
        best_iou_t = mean_iou
        best_t = t
    print(f"  {t:>10.1f}  {mean_iou:>10.4f}{marker}")

# ─────────────────────────────────────────────
# DIAGNOSTIC 3: TuSimple official accuracy
# ─────────────────────────────────────────────
print("\n── DIAGNOSTIC 3: TuSimple-style accuracy ──")
print("  (% of GT lane points correctly predicted within 20px tolerance)")

correct = 0
total   = 0
for rec in sample_records:
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_h, img_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    aug   = val_transform(image=img_rgb,
                          mask=np.zeros((img_h, img_w), dtype=np.uint8))
    img_t = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t).squeeze()).cpu().numpy()

    # Scale factors from original to model input size
    sx = IMG_W / img_w
    sy = IMG_H / img_h

    for lane in rec["lanes"]:
        for x_orig, y_orig in zip(lane, rec["h_samples"]):
            if x_orig == -2:
                continue
            # Map GT point to resized coords
            x_r = int(x_orig * sx)
            y_r = int(y_orig * sy)
            x_r = np.clip(x_r, 0, IMG_W - 1)
            y_r = np.clip(y_r, 0, IMG_H - 1)

            # Check if any pixel within 20px horizontal window is predicted
            x_lo = max(0, x_r - 20)
            x_hi = min(IMG_W, x_r + 20)
            window = prob[y_r, x_lo:x_hi]
            hit = (window > best_t).any()
            correct += int(hit)
            total   += 1

tusimple_acc = correct / total if total > 0 else 0
print(f"  GT points evaluated : {total}")
print(f"  Correctly predicted : {correct}")
print(f"  TuSimple accuracy   : {tusimple_acc*100:.2f}%")
print(f"  (TuSimple SOTA is ~96%, good threshold ≥ 90%)")

# ─────────────────────────────────────────────
# DIAGNOSTIC 4: Visual predictions on 6 frames
# ─────────────────────────────────────────────
print("\n── DIAGNOSTIC 4: Saving visual predictions ──")
fig = plt.figure(figsize=(20, 12))
gs  = gridspec.GridSpec(3, 6, figure=fig)

viz_records = records[:6]
for i, rec in enumerate(viz_records):
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_h, img_w = img_bgr.shape[:2]
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_gt  = make_mask(rec["lanes"], rec["h_samples"], img_h, img_w)

    aug    = val_transform(image=img_rgb, mask=mask_gt)
    img_t  = aug["image"].unsqueeze(0).to(DEVICE)
    mask_t = aug["mask"].numpy()

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t).squeeze()).cpu().numpy()

    pred_mask = (prob > best_t).astype(np.uint8)

    # Resize original image to model size for display
    img_disp = cv2.resize(img_rgb, (IMG_W, IMG_H))

    # Row 1: original
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img_disp)
    ax.set_title(f"Frame {i+1}\nOriginal", fontsize=8)
    ax.axis("off")

    # Row 2: GT mask overlaid
    ax = fig.add_subplot(gs[1, i])
    overlay = img_disp.copy()
    overlay[mask_t == 1] = [0, 255, 0]
    ax.imshow(overlay)
    ax.set_title("GT (green)", fontsize=8)
    ax.axis("off")

    # Row 3: prediction overlaid
    ax = fig.add_subplot(gs[2, i])
    overlay2 = img_disp.copy()
    overlay2[pred_mask == 1] = [255, 100, 0]
    ax.imshow(overlay2)
    iou_val = thresh_ious[best_t][i] if i < len(thresh_ious[best_t]) else 0
    ax.set_title(f"Pred (orange)\nIoU={iou_val:.3f}", fontsize=8)
    ax.axis("off")

plt.suptitle(f"Model Predictions vs GT  |  threshold={best_t}  |  TuSimple acc={tusimple_acc*100:.1f}%",
             fontsize=12)
plt.tight_layout()
out_path = os.path.expanduser("~/lane_model_v2/diagnostics.png")
plt.savefig(out_path, dpi=150)
print(f"  Saved to: {out_path}")
plt.show()

print("\n── SUMMARY ──")
print(f"  Lane pixel ratio  : {ratios.mean()*100:.2f}% (class imbalance {(1-ratios.mean())/ratios.mean():.0f}:1)")
print(f"  Best IoU threshold: {best_t}  →  IoU={best_iou_t:.4f}")
print(f"  TuSimple accuracy : {tusimple_acc*100:.2f}%")
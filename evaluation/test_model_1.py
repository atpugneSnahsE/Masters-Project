"""
Quick Model Test — VIL-100 fine-tuned U-Net on road video
Shows: Original | Lane Mask | Overlay for 12 frames

Usage:
    python3 test_model.py
"""

import os
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = os.path.expanduser("~/Downloads/vil100_model/best_model.pth")
VIDEO_PATH  = os.path.expanduser("~/Downloads/road_clip2.mp4")
OUTPUT_IMG  = os.path.expanduser("~/Downloads/vil100_model/test_results_finetuned.png")

IMG_H       = 368
IMG_W       = 640
THRESHOLD   = 0.36
DEVICE      = "cpu"
NUM_FRAMES  = 12

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
model.eval()
print(f"Model loaded: {MODEL_PATH}")
print(f"Expected IoU: 0.7836")

transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std =(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# EXTRACT FRAMES
# ─────────────────────────────────────────────
cap   = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total} frames | {fps:.1f} fps | {total/fps:.1f}s")

indices = np.linspace(int(fps*10), int(fps*280), NUM_FRAMES, dtype=int)
frames  = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()
print(f"Extracted {len(frames)} frames — running inference...")

# ─────────────────────────────────────────────
# RUN INFERENCE + PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(len(frames), 3, figsize=(15, len(frames) * 3))

axes[0][0].set_title("Original Frame",  fontsize=11, fontweight="bold")
axes[0][1].set_title("Lane Mask",       fontsize=11, fontweight="bold")
axes[0][2].set_title("Overlay (green)", fontsize=11, fontweight="bold")

coverages = []

for i, frame_bgr in enumerate(frames):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    aug   = transform(image=frame_rgb)
    img_t = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t).squeeze()).cpu().numpy()

    mask = (prob > THRESHOLD).astype(np.uint8)

    orig = cv2.resize(frame_rgb, (IMG_W, IMG_H))

    # Overlay: green on lane pixels
    overlay = orig.copy().astype(np.float32)
    overlay[mask == 1] = (
        overlay[mask == 1] * 0.35 +
        np.array([0, 220, 80], dtype=np.float32) * 0.65
    )
    overlay = overlay.astype(np.uint8)

    coverage   = mask.mean() * 100
    confidence = float(prob[mask == 1].mean()) if mask.sum() > 0 else 0.0
    coverages.append(coverage)

    axes[i][0].imshow(orig)
    axes[i][0].axis("off")
    axes[i][0].set_ylabel(f"Frame {i+1}", fontsize=8)

    axes[i][1].imshow(mask, cmap="gray")
    axes[i][1].axis("off")

    axes[i][2].imshow(overlay)
    axes[i][2].axis("off")
    axes[i][2].set_title(
        f"coverage={coverage:.1f}%  avg_conf={confidence:.2f}",
        fontsize=7)

    print(f"  Frame {i+1:2d}: coverage={coverage:.1f}%  conf={confidence:.2f}")

print(f"\nMean coverage across frames: {np.mean(coverages):.1f}%")

plt.suptitle(
    f"Fine-tuned Model (IoU=0.7836) — threshold={THRESHOLD}  "
    f"mean_coverage={np.mean(coverages):.1f}%",
    fontsize=12, fontweight="bold")
plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
plt.savefig(OUTPUT_IMG, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_IMG}")
plt.show()
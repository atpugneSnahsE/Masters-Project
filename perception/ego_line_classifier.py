"""
Ego Lane Classification — Step 3
Uses the trained U-Net (ResNet34) lane detection model to:
  1. Predict lane masks on TuSimple images
  2. Extract individual lane instances from the mask
  3. Determine which lane the ego vehicle is currently in
  4. Visualize results with lane numbering and ego lane highlight

Usage:
    python3 ego_lane_classifier.py
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
import matplotlib.patches as mpatches
from scipy import ndimage

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/archive/TUSimple/train_set")
MODEL_PATH   = os.path.expanduser("~/lane_model_v2/best_model.pth")
LABEL_FILE   = os.path.join(DATASET_ROOT, "label_data_0313.json")
IMG_H, IMG_W = 368, 640
THRESHOLD    = 0.6
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES   = 12   # number of frames to visualize
SAVE_DIR     = os.path.expanduser("~/lane_model_v2")

# ─────────────────────────────────────────────
# LANE INSTANCE EXTRACTION
# ─────────────────────────────────────────────
def extract_lane_instances(mask, min_pixels=50):
    """
    Extract individual lane instances from a binary mask.
    Strategy:
      - Skeletonize mask into thin lines
      - Label connected components
      - Sort lanes left to right by their bottom x position
    Returns list of (label_mask, bottom_x) sorted left to right.
    """
    # Thin the mask to get clean lane centrelines
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thinned  = cv2.ximgproc.thinning(mask * 255) if hasattr(cv2, 'ximgproc') \
               else mask  # fallback if ximgproc not available

    # Label connected components
    labeled, num_features = ndimage.label(thinned)

    lanes = []
    for label_id in range(1, num_features + 1):
        component = (labeled == label_id).astype(np.uint8)
        if component.sum() < min_pixels:
            continue  # skip noise

        # Find bottom-most point of this lane
        rows, cols = np.where(component > 0)
        if len(rows) == 0:
            continue
        bottom_row_idx = np.argmax(rows)
        bottom_x = cols[bottom_row_idx]
        bottom_y = rows[bottom_row_idx]

        lanes.append({
            "mask"    : component,
            "bottom_x": int(bottom_x),
            "bottom_y": int(bottom_y),
            "pixels"  : int(component.sum()),
        })

    # Sort lanes left to right by bottom x position
    lanes.sort(key=lambda l: l["bottom_x"])
    return lanes


def extract_lanes_by_columns(mask, img_w, num_slices=5):
    """
    Alternative lane extraction: divide lower portion of image
    into vertical columns and find lane centroids per column.
    More robust than connected components for dashed lanes.
    Returns sorted list of lane dicts.
    """
    h, w = mask.shape
    roi_top = int(h * 0.5)   # only look at bottom half

    lanes_raw = []
    # Use horizontal projection — find peaks in column sums
    col_sums = mask[roi_top:, :].sum(axis=0).astype(float)

    # Smooth column sums
    kernel_size = 15
    col_sums_smooth = np.convolve(col_sums,
                                  np.ones(kernel_size)/kernel_size,
                                  mode='same')

    # Find peaks (local maxima above threshold)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(col_sums_smooth,
                              height=col_sums_smooth.max() * 0.15,
                              distance=30)

    for peak_x in peaks:
        # Extract a narrow vertical strip around this peak
        x_lo = max(0, peak_x - 20)
        x_hi = min(w, peak_x + 20)
        strip = mask[:, x_lo:x_hi]

        # Find bottom-most pixel in strip
        rows, cols = np.where(strip > 0)
        if len(rows) == 0:
            continue
        bottom_idx = np.argmax(rows)
        bottom_y   = rows[bottom_idx]
        bottom_x   = x_lo + cols[bottom_idx]

        # Build a representative mask for this lane (strip only)
        lane_mask = np.zeros_like(mask)
        lane_mask[:, x_lo:x_hi] = strip

        lanes_raw.append({
            "mask"    : lane_mask,
            "bottom_x": int(bottom_x),
            "bottom_y": int(bottom_y),
            "peak_x"  : int(peak_x),
            "pixels"  : int(strip.sum()),
        })

    # Sort left to right
    lanes_raw.sort(key=lambda l: l["peak_x"])
    return lanes_raw


def identify_ego_lane(lanes, img_w):
    """
    Determine ego lane boundaries.
    The ego lane is bounded by:
      - The rightmost lane whose bottom_x < img_w/2  (left boundary)
      - The leftmost lane whose bottom_x >= img_w/2  (right boundary)
    Returns (ego_left_idx, ego_right_idx, lane_position_string)
    """
    cx = img_w / 2

    left_candidates  = [(i, l) for i, l in enumerate(lanes)
                        if l["bottom_x"] < cx]
    right_candidates = [(i, l) for i, l in enumerate(lanes)
                        if l["bottom_x"] >= cx]

    ego_left_idx  = left_candidates[-1][0]  if left_candidates  else None
    ego_right_idx = right_candidates[0][0]  if right_candidates else None

    # Determine lane position label
    total_lanes = len(lanes)
    if ego_left_idx is not None and ego_right_idx is not None:
        lane_num = ego_left_idx + 1   # 1-indexed from left
        position = f"Lane {lane_num} of {total_lanes - 1}"
    elif ego_left_idx is None:
        position = "Leftmost lane"
    elif ego_right_idx is None:
        position = "Rightmost lane"
    else:
        position = "Unknown"

    return ego_left_idx, ego_right_idx, position


# ─────────────────────────────────────────────
# MODEL SETUP
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
print(f"Model loaded: {MODEL_PATH}")
print(f"Device: {DEVICE}")

val_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────
# LOAD RECORDS
# ─────────────────────────────────────────────
records = []
with open(LABEL_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

# Sample evenly across the dataset for variety
indices = np.linspace(0, len(records) - 1, NUM_FRAMES, dtype=int)
sample_records = [records[i] for i in indices]
print(f"Processing {NUM_FRAMES} frames...")

# ─────────────────────────────────────────────
# LANE COLOURS
# ─────────────────────────────────────────────
LANE_PALETTE = [
    (255,  80,  80),   # red
    ( 80, 200,  80),   # green
    ( 80, 130, 255),   # blue
    (255, 200,  50),   # yellow
    (200,  80, 255),   # purple
    ( 50, 220, 220),   # cyan
]
EGO_LEFT_COLOR  = (255, 255,  50)   # bright yellow  — left ego boundary
EGO_RIGHT_COLOR = ( 50, 255, 255)   # bright cyan    — right ego boundary
EGO_FILL_ALPHA  = 0.20              # transparency of ego lane fill

# ─────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(22, 14))
axes = axes.flatten()

for frame_idx, rec in enumerate(sample_records):
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  Could not read: {img_path}")
        continue

    img_h, img_w = img_bgr.shape[:2]
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Run inference ──
    aug   = val_transform(image=img_rgb,
                          mask=np.zeros((img_h, img_w), dtype=np.uint8))
    img_t = aug["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_t).squeeze()).cpu().numpy()

    pred_mask = (prob > THRESHOLD).astype(np.uint8)

    # ── Extract lane instances ──
    lanes = extract_lanes_by_columns(pred_mask, IMG_W)

    # ── Identify ego lane ──
    ego_left_idx, ego_right_idx, position = identify_ego_lane(lanes, IMG_W)

    # ── Build visualization ──
    vis = cv2.resize(img_rgb, (IMG_W, IMG_H)).copy().astype(np.float32)

    # Draw ego lane fill (semi-transparent)
    if ego_left_idx is not None and ego_right_idx is not None:
        left_mask  = lanes[ego_left_idx]["mask"]
        right_mask = lanes[ego_right_idx]["mask"]

        # Create ego region between the two boundary lanes
        # Find leftmost extent of right lane and rightmost of left lane per row
        ego_fill = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        for row in range(IMG_H):
            left_cols  = np.where(left_mask[row]  > 0)[0]
            right_cols = np.where(right_mask[row] > 0)[0]
            if len(left_cols) > 0 and len(right_cols) > 0:
                x_start = int(left_cols.mean())
                x_end   = int(right_cols.mean())
                if x_end > x_start:
                    ego_fill[row, x_start:x_end] = 1

        fill_color = np.array([100, 220, 100], dtype=np.float32)
        vis[ego_fill == 1] = (vis[ego_fill == 1] * (1 - EGO_FILL_ALPHA) +
                              fill_color * EGO_FILL_ALPHA)

    vis = vis.astype(np.uint8)

    # Draw individual lanes with colours
    for lane_idx, lane in enumerate(lanes):
        color = LANE_PALETTE[lane_idx % len(LANE_PALETTE)]

        # Highlight ego boundaries differently
        if lane_idx == ego_left_idx:
            color     = EGO_LEFT_COLOR
            thickness = 3
        elif lane_idx == ego_right_idx:
            color     = EGO_RIGHT_COLOR
            thickness = 3
        else:
            thickness = 2

        # Draw the lane mask as a coloured overlay
        lane_pixels = lane["mask"] > 0
        vis[lane_pixels] = color

        # Lane label at bottom of lane
        bx = lane["bottom_x"]
        by = min(lane["bottom_y"] + 15, IMG_H - 5)
        label = f"L{lane_idx+1}"
        cv2.putText(vis, label, (max(0, bx-10), by),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Ego lane text overlay
    cv2.rectangle(vis, (0, 0), (IMG_W, 36), (0, 0, 0), -1)
    cv2.putText(vis,
                f"Ego: {position}  |  {len(lanes)} lanes detected",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)

    # ── Plot ──
    ax = axes[frame_idx]
    ax.imshow(vis)
    ax.set_title(f"Frame {frame_idx+1}  —  {position}", fontsize=9)
    ax.axis("off")

    print(f"  Frame {frame_idx+1:2d}: {len(lanes)} lanes  |  ego={position}  |  {rec['raw_file']}")

# Legend
legend_elements = [
    mpatches.Patch(color=np.array(EGO_LEFT_COLOR)/255,  label="Ego left boundary"),
    mpatches.Patch(color=np.array(EGO_RIGHT_COLOR)/255, label="Ego right boundary"),
    mpatches.Patch(color=(0.4, 0.86, 0.4),              label="Ego lane fill"),
    mpatches.Patch(color=(1.0, 0.31, 0.31),             label="Other lanes"),
]
fig.legend(handles=legend_elements, loc="lower center",
           ncol=4, fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

plt.suptitle("Ego Lane Classification  —  U-Net (ResNet34) on TuSimple",
             fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0.05, 1, 1])

out_path = os.path.join(SAVE_DIR, "ego_lane_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {out_path}")
plt.show()
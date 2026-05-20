"""
Lane Marking Type Classifier v2 — Fixed
Fixes:
  1. Strict peak filtering — max 5 lanes, minimum peak height
  2. Along-lane dash analysis — follows actual lane angle instead of vertical
  3. Better rhythm detection using perspective-corrected gap ratios
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
from scipy.signal import find_peaks

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/Downloads/archive/TUSimple/train_set")
MODEL_PATH   = os.path.expanduser("~/lane_model_v2/best_model.pth")
LABEL_FILE   = os.path.join(DATASET_ROOT, "label_data_0313.json")
IMG_H, IMG_W = 368, 640
THRESHOLD    = 0.6
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES   = 12
SAVE_DIR     = os.path.expanduser("~/lane_model_v2")

# Peak detection — stricter
MAX_LANES         = 5      # TuSimple has max 5 lanes
MIN_PEAK_HEIGHT   = 0.25   # minimum fraction of max column sum to count as lane
MIN_PEAK_DISTANCE = 50     # minimum pixel distance between lanes

# Dash analysis
NUM_SAMPLE_ROWS   = 60     # rows to sample along the lane
ROW_START_FRAC    = 0.40   # start from this image height fraction
ROW_END_FRAC      = 0.95
STRIP_HALF_WIDTH  = 12     # pixels either side of lane centre

# ─────────────────────────────────────────────
# MARKING / RHYTHM DEFINITIONS
# ─────────────────────────────────────────────
MARKING_TYPES = {
    "dashed"      : {"color": (80, 200, 80),   "rule": "Overtaking PERMITTED"},
    "solid"       : {"color": (200, 80, 80),   "rule": "Overtaking NOT permitted"},
    "double_solid": {"color": (220, 50, 50),   "rule": "Overtaking FORBIDDEN both directions"},
    "solid_dashed": {"color": (220, 160, 50),  "rule": "Overtaking from DASHED side only"},
    "unknown"     : {"color": (160, 160, 160), "rule": "Marking unclear"},
}
RHYTHM_STATES = {
    "equal"    : {"color": (50, 200, 50),  "advisory": "Maintain speed"},
    "shrinking": {"color": (220, 100, 50), "advisory": "⚠ Slow down — hazard zone ahead"},
    "growing"  : {"color": (50, 150, 220), "advisory": "Leaving hazard zone"},
    "solid"    : {"color": (200, 80, 80),  "advisory": "Solid line — no overtake"},
    "unknown"  : {"color": (160, 160, 160),"advisory": "Insufficient data"},
}


# ─────────────────────────────────────────────
# LANE PEAK DETECTION — strict version
# ─────────────────────────────────────────────
def find_lane_peaks_strict(mask, img_w, img_h):
    """
    Find lane centre x-positions.
    Uses bottom 55% of image, strong smoothing, strict height threshold.
    Returns at most MAX_LANES peaks sorted left to right.
    """
    roi_top  = int(img_h * 0.45)
    col_sums = mask[roi_top:, :].sum(axis=0).astype(float)

    # Smooth heavily to merge adjacent detections from thick masks
    smoothed = np.convolve(col_sums, np.ones(25)/25, mode='same')

    if smoothed.max() == 0:
        return []

    # Strict threshold: must be at least 25% of the strongest lane
    height_thresh = smoothed.max() * MIN_PEAK_HEIGHT

    peaks, props = find_peaks(
        smoothed,
        height=height_thresh,
        distance=MIN_PEAK_DISTANCE,
    )

    if len(peaks) == 0:
        return []

    # Keep only the strongest MAX_LANES peaks
    peak_heights = props["peak_heights"]
    if len(peaks) > MAX_LANES:
        top_idx = np.argsort(peak_heights)[::-1][:MAX_LANES]
        peaks   = peaks[top_idx]

    return sorted(peaks.tolist())


# ─────────────────────────────────────────────
# LANE ANGLE ESTIMATION
# ─────────────────────────────────────────────
def estimate_lane_angle(mask, peak_x, img_h):
    """
    Estimate the angle of a lane by finding its x-centre at multiple
    row levels and fitting a line. Returns (slope, intercept) where
    x = slope * row + intercept.
    """
    row_start = int(img_h * ROW_START_FRAC)
    row_end   = int(img_h * ROW_END_FRAC)
    rows      = np.linspace(row_start, row_end, 20, dtype=int)

    xs, ys = [], []
    for row in rows:
        x_lo = max(0, peak_x - 30)
        x_hi = min(mask.shape[1], peak_x + 30)
        strip = mask[row, x_lo:x_hi]
        if strip.sum() == 0:
            continue
        # centroid x in this row
        cx = x_lo + int(np.average(np.where(strip > 0)[0]))
        xs.append(cx)
        ys.append(row)

    if len(xs) < 4:
        # Fallback: vertical lane
        return 0.0, float(peak_x)

    # Fit line: x = slope * y + intercept
    coeffs = np.polyfit(ys, xs, 1)
    return coeffs[0], coeffs[1]


# ─────────────────────────────────────────────
# ALONG-LANE DASH ANALYSIS
# ─────────────────────────────────────────────
def analyse_along_lane(mask, slope, intercept, img_h):
    """
    Walk along the lane from bottom to top, sampling pixels
    perpendicular to the lane direction. Build a hit sequence
    and classify as dashed/solid + rhythm.

    Returns:
      marking_type : str
      dash_lengths : list[int]
      gap_lengths  : list[int]
      hit_ratio    : float
    """
    row_start = int(img_h * ROW_START_FRAC)
    row_end   = int(img_h * ROW_END_FRAC)
    rows      = np.arange(row_end, row_start, -1)  # bottom to top

    hits = []
    for row in rows:
        cx = int(slope * row + intercept)
        x_lo = max(0, cx - STRIP_HALF_WIDTH)
        x_hi = min(mask.shape[1], cx + STRIP_HALF_WIDTH)
        if x_hi <= x_lo:
            hits.append(0)
            continue
        pixel_count = int(mask[row, x_lo:x_hi].sum())
        hits.append(1 if pixel_count >= 3 else 0)

    hits = np.array(hits)
    if hits.sum() == 0:
        return "unknown", [], [], 0.0

    hit_ratio = hits.mean()

    # Extract runs
    dash_lengths, gap_lengths = [], []
    current = hits[0]
    run = 1
    for i in range(1, len(hits)):
        if hits[i] == current:
            run += 1
        else:
            (dash_lengths if current == 1 else gap_lengths).append(run)
            current = hits[i]
            run = 1
    (dash_lengths if current == 1 else gap_lengths).append(run)

    # Classify
    if hit_ratio > 0.80:
        marking_type = "solid"
    elif hit_ratio < 0.60 and len(gap_lengths) >= 2:
        marking_type = "dashed"
    elif hit_ratio >= 0.60:
        marking_type = "solid"
    else:
        marking_type = "unknown"

    return marking_type, dash_lengths, gap_lengths, hit_ratio


# ─────────────────────────────────────────────
# DASH RHYTHM ANALYSIS
# ─────────────────────────────────────────────
def analyse_rhythm(gap_lengths):
    """
    Analyse gap length trend from bottom (near) to top (far).
    In perspective, gaps naturally shrink toward the horizon.
    We measure the RATE of shrinkage:
      - If shrinking faster than expected → hazard zone
      - If roughly constant rate         → normal
      - If growing                       → leaving hazard zone

    Uses ratio of consecutive gaps rather than absolute differences
    to account for natural perspective compression.
    """
    if len(gap_lengths) < 3:
        return "unknown"

    gaps = np.array(gap_lengths, dtype=float)

    # Compute gap ratios (each gap / previous gap)
    # In normal perspective, ratio < 1.0 (gaps shrink toward horizon)
    ratios = gaps[1:] / (gaps[:-1] + 1e-6)

    mean_ratio = ratios.mean()

    # Normal perspective: ratios around 0.85-1.0
    # Hazard zone: ratios well below 0.75 (extra shrinkage)
    # Growing: ratios above 1.05
    if mean_ratio < 0.75:
        return "shrinking"
    elif mean_ratio > 1.05:
        return "growing"
    else:
        return "equal"


def classify_lane_pair(left_type, right_type):
    if left_type == "solid" and right_type == "solid":
        return "double_solid"
    elif left_type == "dashed" and right_type == "solid":
        return "solid_dashed"
    elif left_type == "solid" and right_type == "dashed":
        return "solid_dashed"
    elif left_type == "dashed" and right_type == "dashed":
        return "dashed"
    elif left_type == "solid" or right_type == "solid":
        return "solid"
    else:
        return "unknown"


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
print(f"Model   : U-Net ResNet34  |  {MODEL_PATH}")
print(f"Device  : {DEVICE}\n")

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

indices        = np.linspace(0, len(records)-1, NUM_FRAMES, dtype=int)
sample_records = [records[i] for i in indices]
print(f"Processing {NUM_FRAMES} frames...\n")


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(24, 16))
axes = axes.flatten()

for frame_idx, rec in enumerate(sample_records):
    img_path = os.path.join(DATASET_ROOT, rec["raw_file"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        continue

    img_h, img_w = img_bgr.shape[:2]
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── Inference ──
    aug   = val_transform(image=img_rgb,
                          mask=np.zeros((img_h, img_w), dtype=np.uint8))
    img_t = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.sigmoid(model(img_t).squeeze()).cpu().numpy()
    pred_mask = (prob > THRESHOLD).astype(np.uint8)

    # ── Find lane peaks (strict) ──
    peaks = find_lane_peaks_strict(pred_mask, IMG_W, IMG_H)
    if len(peaks) < 2:
        axes[frame_idx].imshow(cv2.resize(img_rgb, (IMG_W, IMG_H)))
        axes[frame_idx].set_title(f"Frame {frame_idx+1}\nInsufficient lanes", fontsize=8)
        axes[frame_idx].axis("off")
        continue

    # ── Analyse each lane ──
    lane_results = []
    for peak_x in peaks:
        slope, intercept = estimate_lane_angle(pred_mask, peak_x, IMG_H)
        m_type, dashes, gaps, hit_ratio = analyse_along_lane(
            pred_mask, slope, intercept, IMG_H)
        rhythm = analyse_rhythm(gaps) if m_type == "dashed" else \
                 ("solid" if m_type == "solid" else "unknown")
        lane_results.append({
            "peak_x"   : peak_x,
            "slope"    : slope,
            "intercept": intercept,
            "type"     : m_type,
            "rhythm"   : rhythm,
            "dashes"   : dashes,
            "gaps"     : gaps,
            "hit_ratio": hit_ratio,
        })

    # ── Ego lane ──
    cx = IMG_W / 2
    left_results  = [l for l in lane_results if l["peak_x"] < cx]
    right_results = [l for l in lane_results if l["peak_x"] >= cx]
    ego_left  = left_results[-1]  if left_results  else None
    ego_right = right_results[0]  if right_results else None

    combined_type = classify_lane_pair(
        ego_left["type"]  if ego_left  else "unknown",
        ego_right["type"] if ego_right else "unknown",
    )

    # Pick best rhythm reading
    ego_rhythm = "unknown"
    for candidate in [ego_left, ego_right]:
        if candidate and candidate["rhythm"] not in ("solid", "unknown"):
            ego_rhythm = candidate["rhythm"]
            break
    if ego_rhythm == "unknown":
        for candidate in [ego_left, ego_right]:
            if candidate and candidate["rhythm"] != "unknown":
                ego_rhythm = candidate["rhythm"]
                break

    marking_info = MARKING_TYPES.get(combined_type, MARKING_TYPES["unknown"])
    rhythm_info  = RHYTHM_STATES.get(ego_rhythm,    RHYTHM_STATES["unknown"])

    # ── Visualization ──
    vis = cv2.resize(img_rgb, (IMG_W, IMG_H)).copy()

    # Draw ego lane fill
    if ego_left and ego_right:
        fill_color = np.array(marking_info["color"], dtype=np.float32)
        fill       = np.zeros_like(vis, dtype=np.float32)
        roi_top    = int(IMG_H * 0.45)
        lx = ego_left["peak_x"]
        rx = ego_right["peak_x"]
        fill[roi_top:, lx:rx] = fill_color
        vis = cv2.addWeighted(
            vis.astype(np.float32), 1.0,
            fill, 0.20, 0).astype(np.uint8)

    # Draw each detected lane along its fitted angle
    row_start = int(IMG_H * ROW_START_FRAC)
    row_end   = int(IMG_H * ROW_END_FRAC)

    for lr in lane_results:
        is_ego = (lr is ego_left or lr is ego_right)
        color  = MARKING_TYPES.get(lr["type"], MARKING_TYPES["unknown"])["color"]
        thick  = 3 if is_ego else 1

        # Draw fitted lane line
        rows  = np.linspace(row_start, row_end, 30, dtype=int)
        pts   = [(int(lr["slope"]*r + lr["intercept"]), int(r)) for r in rows]
        pts   = [(max(0, min(IMG_W-1, x)), y) for x, y in pts]

        # Draw as dashed or solid pattern on visualization
        for i in range(0, len(pts)-1, 2 if lr["type"] == "dashed" else 1):
            if i+1 < len(pts):
                cv2.line(vis, pts[i], pts[i+1], color, thick)

        # Label at bottom
        bx = int(lr["slope"] * row_end + lr["intercept"])
        bx = max(5, min(IMG_W-25, bx))
        cv2.putText(vis, lr["type"][:3].upper(),
                    (bx-8, row_end + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # Info bar
    cv2.rectangle(vis, (0, 0), (IMG_W, 58), (0, 0, 0), -1)
    line1 = f"Marking: {combined_type.replace('_',' ').upper()}  |  {marking_info['rule']}"
    line2 = f"Rhythm : {ego_rhythm.upper()}  |  {rhythm_info['advisory']}"
    cv2.putText(vis, line1, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, marking_info["color"], 1)
    cv2.putText(vis, line2, (6, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, rhythm_info["color"], 1)

    axes[frame_idx].imshow(vis)
    axes[frame_idx].set_title(
        f"Frame {frame_idx+1}  |  {combined_type}  |  {ego_rhythm}", fontsize=8)
    axes[frame_idx].axis("off")

    # Terminal
    print(f"Frame {frame_idx+1:2d}: {len(peaks)} lanes")
    print(f"  Ego left  x={ego_left['peak_x'] if ego_left else 'n/a':>3}  "
          f"type={ego_left['type'] if ego_left else 'n/a':<8}  "
          f"hit={ego_left['hit_ratio']:.2f}  "
          f"gaps={ego_left['gaps']}" if ego_left else "  Ego left : n/a")
    print(f"  Ego right x={ego_right['peak_x'] if ego_right else 'n/a':>3}  "
          f"type={ego_right['type'] if ego_right else 'n/a':<8}  "
          f"hit={ego_right['hit_ratio']:.2f}  "
          f"gaps={ego_right['gaps']}" if ego_right else "  Ego right: n/a")
    print(f"  → {combined_type}  |  rhythm={ego_rhythm}  |  "
          f"{marking_info['rule']}  |  {rhythm_info['advisory']}\n")


# ─────────────────────────────────────────────
# LEGEND + SAVE
# ─────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=np.array(MARKING_TYPES["dashed"]["color"])/255,
                   label="Dashed — overtake OK"),
    mpatches.Patch(color=np.array(MARKING_TYPES["solid"]["color"])/255,
                   label="Solid — no overtake"),
    mpatches.Patch(color=np.array(MARKING_TYPES["double_solid"]["color"])/255,
                   label="Double solid — forbidden"),
    mpatches.Patch(color=np.array(MARKING_TYPES["solid_dashed"]["color"])/255,
                   label="Solid+Dashed — one side only"),
    mpatches.Patch(color=np.array(RHYTHM_STATES["equal"]["color"])/255,
                   label="Rhythm: equal — maintain speed"),
    mpatches.Patch(color=np.array(RHYTHM_STATES["shrinking"]["color"])/255,
                   label="Rhythm: shrinking — slow down"),
    mpatches.Patch(color=np.array(RHYTHM_STATES["growing"]["color"])/255,
                   label="Rhythm: growing — leaving hazard"),
]
fig.legend(handles=legend_patches, loc="lower center",
           ncol=4, fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.005))

plt.suptitle(
    "Lane Marking Type Classifier v2 + Dash Rhythm Analyser\n"
    "Rule-based | Along-lane analysis | EU Road Marking Rules",
    fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.06, 1, 1])

out_path = os.path.join(SAVE_DIR, "marking_classifier_v2_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to: {out_path}")
plt.show()
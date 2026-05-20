import timm.models.regnet
if not hasattr(timm.models.regnet, "RegNetCfg"):
    class RegNetCfg:
        pass
    timm.models.regnet.RegNetCfg = RegNetCfg

# Force non-interactive Agg backend BEFORE importing pyplot.
# matplotlib's default Tkinter backend crashes when plt.savefig() is called
# from a background thread (the CARLA camera callback). Agg is a pure
# file renderer with no GUI and no thread restrictions.
import matplotlib
matplotlib.use("Agg")

import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
import os

os.makedirs("xai_output", exist_ok=True)

# ====================== MODEL ======================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load("lane_model.pth", map_location="cpu"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

transform = T.Compose([T.ToPILImage(), T.Resize((256, 512)), T.ToTensor()])

# ====================== LIME ======================
def batch_predict(images):
    model.eval()
    processed = []
    for img in images:
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        processed.append(transform(img))
    batch = torch.stack(processed).to(device)
    with torch.no_grad():
        preds = model(batch)
        probs = torch.sigmoid(preds).cpu().numpy()
    return probs.reshape(len(images), -1)


# ====================== XAI HELPERS ======================

def fill_lane_corridor(line_mask, h, w):
    """
    Fill the drivable corridor between the two detected lane lines.

    Blob-centroid split (replaces the naive image-midpoint split)
    ─────────────────────────────────────────────────────────────
    Instead of assuming the left line is always left of w//2, we find the
    two largest connected blobs in line_mask and label them LEFT / RIGHT by
    their column centroid.  This handles curves and intersections where both
    lines can appear on the same side of the image.

    Anti-bleed rules
    ────────────────
    1. Both blobs present  → fill between their inner edges, row by row.
    2. Only one blob       → project the missing edge using ref_width
                             (median of measured widths, or 35 % of w).
    3. Per-row hard cap    → corridor clamped to [MIN, MAX] pixels wide.
    4. Final blob guard    → drops any filled region wider than MAX_LANE_PX.
    """
    MIN_LANE_PX     = 60
    MAX_LANE_PX     = int(w * 0.55)
    DEFAULT_LANE_PX = int(w * 0.35)

    corridor = np.zeros((h, w), dtype=np.uint8)

    # ── Identify the two main line blobs by centroid ───────────────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_mask)
    if n_labels < 2:
        return corridor   # nothing detected at all

    # Sort non-background components by area, keep top-2
    # but only consider blobs whose centroid is in the lower 60 % of the
    # image — real lane lines seen from a forward camera are always there.
    # Blobs with centroids higher up are guardrails, painted walls, etc.
    CENTROID_Y_MIN = h * 0.40   # centroid must be below this fraction

    road_comps = [i for i in range(1, n_labels)
                  if centroids[i][1] >= CENTROID_Y_MIN]
    road_comps = sorted(road_comps,
                        key=lambda i: stats[i, cv2.CC_STAT_AREA],
                        reverse=True)[:2]

    comp_ids = road_comps

    if len(comp_ids) == 1:
        # Only one blob — label it by centroid position
        cx = centroids[comp_ids[0]][0]
        if cx < w * 0.5:
            left_id, right_id = comp_ids[0], None
        else:
            left_id, right_id = None, comp_ids[0]
    else:
        # Two blobs — assign left/right by centroid x
        c0 = centroids[comp_ids[0]][0]
        c1 = centroids[comp_ids[1]][0]
        if c0 <= c1:
            left_id, right_id = comp_ids[0], comp_ids[1]
        else:
            left_id, right_id = comp_ids[1], comp_ids[0]

    def blob_mask(bid):
        return (labels == bid).astype(np.uint8) * 255 if bid is not None else None

    left_blob  = blob_mask(left_id)
    right_blob = blob_mask(right_id)

    # ── Pass 1: measure ref_width from rows with both blobs ────────────────
    lane_widths = []
    row_bounds  = {}   # y → (xl, xr)

    for y in range(int(h * 0.40), h):
        if left_blob is not None and right_blob is not None:
            lx = np.where(left_blob[y]  > 0)[0]
            rx = np.where(right_blob[y] > 0)[0]
            if len(lx) > 2 and len(rx) > 2:
                xl = int(np.max(lx))
                xr = int(np.min(rx))
                if MIN_LANE_PX < (xr - xl) < MAX_LANE_PX:
                    row_bounds[y] = (xl, xr)
                    lane_widths.append(xr - xl)

    ref_width = (int(np.median(lane_widths))
                 if len(lane_widths) >= 3
                 else DEFAULT_LANE_PX)

    # ── Pass 2: fill ───────────────────────────────────────────────────────
    for y in range(int(h * 0.40), h):
        if y in row_bounds:
            xl, xr = row_bounds[y]
        else:
            # Single-blob fallback using ref_width
            lx = np.where(left_blob[y]  > 0)[0] if left_blob  is not None else []
            rx = np.where(right_blob[y] > 0)[0] if right_blob is not None else []

            if len(lx) > 2 and len(rx) == 0:
                xl = int(np.max(lx))
                xr = min(xl + ref_width, w - 1)
            elif len(rx) > 2 and len(lx) == 0:
                xr = int(np.min(rx))
                xl = max(xr - ref_width, 0)
            else:
                continue

        # Hard clamp
        if (xr - xl) > MAX_LANE_PX:
            centre = (xl + xr) // 2
            xl = centre - MAX_LANE_PX // 2
            xr = centre + MAX_LANE_PX // 2
        if (xr - xl) >= MIN_LANE_PX:
            corridor[y, xl:xr + 1] = 255

    # ── Morphological clean-up ─────────────────────────────────────────────
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))

    # ── Final blob width guard ─────────────────────────────────────────────
    nc, clabels, cstats, _ = cv2.connectedComponentsWithStats(corridor)
    clean = np.zeros_like(corridor)
    for i in range(1, nc):
        if cstats[i, cv2.CC_STAT_WIDTH] <= MAX_LANE_PX:
            clean[clabels == i] = 255
    return clean


def build_road_roi_mask(h, w, horizon_frac=0.42):
    """Trapezoid mask that covers only the road surface below the horizon."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array([
        [0,          h],
        [w,          h],
        [int(w * 0.75), int(h * horizon_frac)],
        [int(w * 0.25), int(h * horizon_frac)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def make_corridor_confidence(prob_vis, corridor_mask, line_mask, h, w):
    """
    Confidence map where:
      - inside the drivable corridor  → the mean prob of the two lane lines
        (high, uniform green — the car knows this is its lane)
      - on the line pixels themselves → raw model probability
      - everywhere else               → raw model probability (mostly red)
    """
    conf = prob_vis.copy()

    # Compute representative line confidence as the mean prob over line pixels
    line_pixels = prob_vis[line_mask > 0]
    line_conf   = float(np.mean(line_pixels)) if len(line_pixels) > 0 else 0.85

    # Fill corridor with that uniform confidence value
    conf[corridor_mask > 0] = line_conf

    # Keep the actual line pixels at their raw (usually higher) probability
    conf[line_mask > 0] = prob_vis[line_mask > 0]

    # Smooth so there are no hard edges between corridor and surroundings
    conf = cv2.GaussianBlur(conf, (21, 21), 0)

    # Re-suppress sky
    conf[:int(h * 0.35), :] = 0
    return conf


# ====================== FIXED XAI ======================
def generate_xai_figure(rgb_image, frame_id):
    save_path = f"xai_output/xai_frame_{frame_id:06d}.png"
    h, w = rgb_image.shape[:2]

    inp = transform(rgb_image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)
    prob = torch.sigmoid(pred).cpu().squeeze().numpy()
    prob_vis = cv2.resize(prob, (w, h), cv2.INTER_LINEAR)

    # Suppress sky
    prob_vis[:int(h * 0.38), :] = 0

    # Match main pipeline threshold
    lower_half = prob_vis[int(h * 0.4):, :]
    thresh     = max(0.22, np.percentile(lower_half, 78))

    line_mask = (prob_vis > thresh).astype(np.uint8) * 255
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((11, 9), np.uint8))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN,  np.ones((7,  7), np.uint8))

    # Keep only the two largest blobs (= left + right lane line)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_mask)
    if n_labels > 2:
        # Sort components by area descending, keep top-2
        areas   = stats[1:, cv2.CC_STAT_AREA]
        top2    = np.argsort(areas)[::-1][:2] + 1   # +1 because label 0 = background
        line_mask = np.zeros_like(line_mask)
        for lbl in top2:
            line_mask[labels == lbl] = 255

    print(f"   Frame {frame_id} | Thresh: {thresh:.3f} | Line px: {(line_mask > 0).sum()}")

    # ── CORRIDOR FILL ──────────────────────────────────────────────────────────
    corridor = fill_lane_corridor(line_mask, h, w)

    # Visualisation: green corridor + brighter green on the lines themselves
    enhanced = rgb_image.copy()
    green_corridor = corridor.copy()
    green_corridor[line_mask > 0] = 0          # lines handled separately
    enhanced[green_corridor > 0]  = [30, 200, 80]   # corridor: semi-transparent look
    # Blend corridor so road texture shows through slightly
    corridor_overlay = rgb_image.copy()
    corridor_overlay[green_corridor > 0] = [30, 200, 80]
    enhanced = cv2.addWeighted(corridor_overlay, 0.55, rgb_image.copy(), 0.45, 0)
    enhanced[line_mask > 0] = [80, 255, 140]   # line pixels: bright green on top

    # ── LIME (road-ROI masked) ─────────────────────────────────────────────────
    # Black-out everything above the horizon so LIME superpixels never include sky
    road_roi  = build_road_roi_mask(h, w, horizon_frac=0.42)
    lime_input = rgb_image.copy().astype(np.float64) / 255.0
    lime_input[road_roi == 0] = 0.0            # zero out sky/walls for LIME

    explainer   = lime_image.LimeImageExplainer(verbose=False)
    explanation = explainer.explain_instance(
        lime_input,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_features=50,
        num_samples=800,
        batch_size=20,
        random_seed=42
    )
    temp, lime_boundary = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=25, hide_rest=False
    )
    # Restore original colours in the LIME visualisation (only show boundaries)
    lime_vis = mark_boundaries(rgb_image.astype(np.float64) / 255.0, lime_boundary)

    # ── CONFIDENCE MAP ────────────────────────────────────────────────────────
    trust_map = make_corridor_confidence(prob_vis, corridor, line_mask, h, w)

    # ── PLOT ──────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].imshow(rgb_image)
    axs[0].set_title("Input RGB")
    axs[0].axis('off')

    axs[1].imshow(enhanced)
    axs[1].set_title("Drivable Corridor")
    axs[1].axis('off')

    axs[2].imshow(lime_vis)
    axs[2].set_title("LIME Attribution (road ROI)")
    axs[2].axis('off')

    im = axs[3].imshow(trust_map, cmap='RdYlGn', vmin=0, vmax=1)
    axs[3].set_title("Lane Confidence Map")
    axs[3].axis('off')
    plt.colorbar(im, ax=axs[3])

    plt.suptitle(f"XAI Analysis - Frame {frame_id}", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ XAI saved: {save_path}")


# ====================== CARLA SETUP ======================
client = carla.Client("localhost", 2000)
client.set_timeout(10)
world = client.load_world("Town04")

bp = world.get_blueprint_library()
vehicle_bp = bp.filter("vehicle.audi.a2")[0]

vehicle = None
for sp in world.get_map().get_spawn_points():
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle:
        break

vehicle.set_autopilot(True)

cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1024")
cam_bp.set_attribute("image_size_y", "512")

camera = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=1.35, z=1.25), carla.Rotation(pitch=-5)),
    attach_to=vehicle
)

# ====================== STATE & HELPERS ======================
class LaneState(Enum):
    NORMAL         = "NORMAL"
    CROSSWALK      = "CROSSWALK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MERGING        = "MERGING"
    CURVE          = "CURVE"

left_hist  = deque(maxlen=16)
right_hist = deque(maxlen=16)
_left_committed  = None
_right_committed = None
_left_contrary   = 0
_right_contrary  = 0

mask_buf  = deque(maxlen=5)
_lfit_buf = deque(maxlen=12)
_rfit_buf = deque(maxlen=12)

trend_buf   = deque(maxlen=30)
trend_frame = 0

last_gap      = None
last_trend    = None
current_state = LaneState.NORMAL
freeze_count  = 0
prev_left     = "solid"
prev_right    = "solid"

HYSTERESIS_N = 14


class KalmanFilter:
    def __init__(self, process_var=0.1, measurement_var=0.8):
        self.x = 0.0; self.p = 1.0; self.q = process_var; self.r = measurement_var

    def update(self, z):
        self.p += self.q
        k      = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x


kf_gap = KalmanFilter(0.08, 1.2)


def apply_clahe(img):
    lab        = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl         = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)


def carla_to_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
        (image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1]


def get_speed_kmh():
    vel = vehicle.get_velocity()
    return (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5 * 3.6


def filter_horizontal_noise(mask):
    clean = mask.copy()
    h, w  = mask.shape
    for y in range(h):
        xs   = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        span = xs[-1] - xs[0]
        if span > w * 0.58 or len(xs) < 8:
            clean[y] = 0
    return clean


def detect_crosswalk(mask):
    h, w = mask.shape
    roi  = mask[int(h * 0.55):int(h * 0.85), :]
    rows = np.sum(roi > 0, axis=1)
    return bool(np.sum(rows > w * 0.35) > 12)


def bev(mask):
    h, w = mask.shape
    src  = np.float32([[w*0.22, h*0.88], [w*0.78, h*0.88],
                       [w*0.38, h*0.63], [w*0.62, h*0.63]])
    dst  = np.float32([[w*0.25, h],      [w*0.75, h],
                       [w*0.25, 0],      [w*0.75, 0]])
    M    = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(mask, M, (w, h))


def get_line_type(band):
    row_hits = np.sum(band > 0, axis=1) > 2
    occ      = np.mean(row_hits)
    return "solid" if occ > 0.68 else "dashed" if occ > 0.16 else "unknown"


def _weighted_majority(hist):
    real = [x for x, c in hist if x != "unknown"]
    pool = real if len(real) >= 3 else [x for x, c in hist]
    if not pool:
        return "unknown"
    from collections import defaultdict
    scores = defaultdict(float)
    for label, conf in hist:
        if label != "unknown" or len(real) < 3:
            scores[label] += conf
    return max(scores, key=scores.get)


def classify_boundary(side, hist, side_id):
    global _left_committed, _right_committed, _left_contrary, _right_contrary
    h, w         = side.shape
    col_profile  = np.sum(side > 0, axis=0)
    peak_col     = int(np.argmax(col_profile))
    peak_val     = int(col_profile[peak_col])

    if peak_val < 5:
        raw, conf = "unknown", 0.25
    else:
        search   = col_profile.astype(float).copy()
        suppress = max(4, int(w * 0.055))
        search[max(0, peak_col - suppress):min(w, peak_col + suppress)] = 0
        second_col = int(np.argmax(search))
        second_val = int(search[second_col])

        distance             = abs(second_col - peak_col)
        is_double_candidate  = (second_val >= peak_val * 0.55 and
                                 8 <= distance <= int(w * 0.11))

        bh        = max(6, int(w * 0.11))
        main_band = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
        main_type = get_line_type(main_band)

        if is_double_candidate:
            second_band = side[:, max(0, second_col - bh):min(w, second_col + bh)]
            second_type = get_line_type(second_band)
            raw = (f"double {main_type}"
                   if main_type == second_type != "unknown"
                   else main_type)
        else:
            raw = main_type

        conf  = min(1.0, peak_val / 30.0)
        conf *= 0.78 if "solid" in raw else 0.92
        if "double" in raw:
            conf *= 0.88

    hist.append((raw, conf))
    vote = _weighted_majority(hist) if len(hist) >= 5 else raw

    if side_id == 'left':
        committed, contrary = _left_committed, _left_contrary
    else:
        committed, contrary = _right_committed, _right_contrary

    if committed is None:
        committed = vote
        contrary  = 0
    elif vote == committed:
        contrary = 0
    else:
        contrary += 1
        if contrary >= HYSTERESIS_N:
            committed = vote
            contrary  = 0

    if side_id == 'left':
        _left_committed, _left_contrary   = committed, contrary
    else:
        _right_committed, _right_contrary = committed, contrary

    return committed


def compute_lane_width(mask):
    h, w    = mask.shape
    mid     = w // 2
    widths  = []
    for y in range(int(h * 0.6), h, 4):
        row = np.where(mask[y] > 0)[0]
        if len(row) == 0:
            continue
        L = row[row <  mid]
        R = row[row >= mid]
        if len(L) > 3 and len(R) > 3:
            widths.append(np.min(R) - np.max(L))
    return np.median(widths) if widths else 0


def measure_dash_trend(side, speed_kmh):
    global trend_frame
    trend_frame += 1
    h, w        = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col    = int(np.argmax(col_profile))
    if col_profile[peak_col] < 5:
        return None, None

    bh     = max(6, int(w * 0.13))
    band   = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
    signal = (np.sum(band > 0, axis=1) > 2).astype(np.int8)
    diff   = np.diff(signal)
    starts = np.where(diff == 1)[0]  + 1
    ends   = np.where(diff == -1)[0] + 1
    if len(ends) > 0 and len(starts) > 0 and ends[0] < starts[0]:
        ends = ends[1:]

    n = min(len(starts), len(ends))
    if n < 2:
        return None, None

    gaps = [int(starts[i + 1]) - int(ends[i])
            for i in range(n - 1)
            if 5 < (starts[i + 1] - ends[i]) < h * 0.6]
    if not gaps:
        return None, None

    mean_gap = float(np.mean(gaps))
    smoothed = kf_gap.update(mean_gap)
    trend_buf.append((trend_frame, smoothed))

    if len(trend_buf) < 12:
        return smoothed, "calculating..."

    recent = list(trend_buf)[-15:]
    data   = np.array(recent)
    slope  = np.polyfit(data[:, 0], data[:, 1], 1)[0]
    trend  = ("diverging"  if slope >  0.20 else
              "converging" if slope < -0.20 else
              "constant")
    return smoothed, trend


def update_state(lt, rt, gap, crosswalk, lane_width):
    global current_state, freeze_count
    if crosswalk:
        current_state = LaneState.CROSSWALK
        freeze_count  = 12
        return
    if lane_width < 70 or lane_width > 480:
        current_state = LaneState.LOW_CONFIDENCE
        freeze_count  = max(freeze_count, 6)
        return
    if gap is not None and gap > 180 and ("dashed" in lt or "dashed" in rt):
        current_state = LaneState.MERGING
    else:
        current_state = LaneState.NORMAL


def make_lane_label(lt, rt):
    if lt == rt == "unknown":     return "UNKNOWN"
    if lt == "solid" and rt == "solid":   return "DOUBLE SOLID"
    if lt == "dashed" and rt == "dashed": return "DOUBLE DASHED"
    return f"{lt} | {rt}"


def draw_lane_overlay(overlay, mask):
    h, w    = mask.shape
    mid     = w // 2
    l_pts, r_pts = [], []
    for y in range(int(h * 0.62), h, 3):
        row = np.where(mask[y] > 0)[0]
        if len(row) < 8:
            continue
        L = row[row <  mid]
        R = row[row >= mid]
        if len(L) > 5: l_pts.append([float(y), float(np.max(L))])
        if len(R) > 5: r_pts.append([float(y), float(np.min(R))])

    if len(l_pts) >= 8:
        try:
            _lfit_buf.append(np.polyfit(np.array(l_pts)[:, 0],
                                         np.array(l_pts)[:, 1], 2))
        except Exception:
            pass
    if len(r_pts) >= 8:
        try:
            _rfit_buf.append(np.polyfit(np.array(r_pts)[:, 0],
                                         np.array(r_pts)[:, 1], 2))
        except Exception:
            pass

    if len(_lfit_buf) < 3 or len(_rfit_buf) < 3:
        return

    lfit = np.mean(list(_lfit_buf)[-8:], axis=0)
    rfit = np.mean(list(_rfit_buf)[-8:], axis=0)

    ys  = np.arange(int(h * 0.62), h, dtype=np.float32)
    lxs = np.polyval(lfit, ys).clip(0, mid - 1).astype(np.int32)
    rxs = np.polyval(rfit, ys).clip(mid, w - 1).astype(np.int32)
    yi  = ys.astype(np.int32)

    fill = overlay.copy()
    pts  = np.concatenate([
        np.stack([lxs, yi], axis=1),
        np.stack([rxs, yi], axis=1)[::-1]
    ], axis=0)
    cv2.fillPoly(fill, [pts], (0, 180, 0))
    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)

    for i in range(len(yi) - 1):
        cv2.line(overlay, (lxs[i], yi[i]), (lxs[i + 1], yi[i + 1]), (0, 255, 255), 4)
        cv2.line(overlay, (rxs[i], yi[i]), (rxs[i + 1], yi[i + 1]), (0, 255, 255), 4)


def draw_hud(overlay, lane_label, state, speed, gap, trend, width):
    h, w = overlay.shape[:2]
    bar  = np.zeros((145, w, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (w, 145), (0, 0, 0), -1)
    overlay[0:145] = cv2.addWeighted(overlay[0:145], 0.35, bar, 0.65, 0)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(overlay, f"Lane: {lane_label}",
                (30, 45), font, 1.05, (0, 255, 255), 2)
    cv2.putText(overlay, f"State: {state.value} | Speed: {speed:.0f} km/h",
                (30, 80), font, 0.85, (0, 255, 180), 2)

    info = (f"Gap: {gap:.1f}px | Trend: {trend} | Width: {int(width)}px"
            if gap is not None else f"Width: {int(width)}px")
    cv2.putText(overlay, info, (30, 120), font, 0.82, (255, 255, 100), 2)

    gx = w - 380
    gy = h - 60
    cv2.putText(overlay, "L", (gx - 30, gy + 22), font, 0.85, (200, 200, 200), 2)
    cv2.putText(overlay, "R", (gx + 220, gy + 22), font, 0.85, (200, 200, 200), 2)
    cv2.rectangle(overlay, (gx, gy),       (gx + 200, gy + 28), (40,  40,  40),  -1)
    cv2.rectangle(overlay, (gx, gy),       (gx + 200, gy + 28), (255, 255, 255),  2)
    cv2.line(overlay,      (gx + 98, gy),  (gx + 102, gy + 28), (80,  80,  80),   3)

    offset = 0
    if width > 100:
        offset = (width - 300) / 7
    pos = int(gx + 100 + np.clip(offset, -90, 90))
    cv2.rectangle(overlay, (pos - 10, gy - 3), (pos + 10, gy + 31), (0, 180, 255), -1)


# ====================== MAIN PROCESS CALLBACK ======================
latest      = None
frame_count = 0
XAI_EVERY   = 35


def process(image):
    global latest, frame_count, freeze_count, prev_left, prev_right
    global last_gap, last_trend, current_state

    frame_count += 1
    raw_rgb   = carla_to_rgb(image)
    rgb       = apply_clahe(raw_rgb)
    speed_kmh = get_speed_kmh()

    inp = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)

    prob = torch.sigmoid(pred).cpu().squeeze().numpy()

    # Main pipeline threshold (consistent with XAI path after fixes)
    lower_half = prob[int(prob.shape[0] * 0.4):, :]
    thresh     = max(0.22, np.percentile(lower_half, 78))

    mask = (prob > thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((7,  7), np.uint8))
    mask = cv2.resize(mask, (raw_rgb.shape[1], raw_rgb.shape[0]), cv2.INTER_NEAREST)
    mask = filter_horizontal_noise(mask)

    mask_buf.append(mask)
    fused = np.median(np.array(mask_buf), axis=0).astype(np.uint8)

    crosswalk  = detect_crosswalk(fused)
    b          = bev(fused)
    left_bev   = b[:, :b.shape[1] // 2]
    right_bev  = b[:, b.shape[1] // 2:]

    if freeze_count > 0:
        lt, rt       = prev_left, prev_right
        freeze_count -= 1
    else:
        lt = classify_boundary(left_bev,  left_hist,  'left')
        rt = classify_boundary(right_bev, right_hist, 'right')
        prev_left, prev_right = lt, rt

    lane_label = make_lane_label(lt, rt)
    lane_width = compute_lane_width(fused)

    side = (left_bev  if "dashed" in lt else
            right_bev if "dashed" in rt else None)
    gap = trend = None
    if side is not None and not crosswalk:
        gap, trend = measure_dash_trend(side, speed_kmh)
        if gap is not None:
            last_gap, last_trend = gap, trend

    update_state(lt, rt, gap, crosswalk, lane_width)

    if frame_count % XAI_EVERY == 0:
        generate_xai_figure(raw_rgb.copy(), frame_count)

    latest = (raw_rgb, fused, b, lane_label,
              last_gap, last_trend, current_state, lane_width, speed_kmh)


camera.listen(process)

print("🚀 Running... Press ESC to quit")
try:
    while True:
        if latest is not None:
            rgb, mask, b, lane_label, gap, trend, state, width, speed = latest
            overlay = rgb.copy()

            overlay[mask == 255] = [0, 255, 120]
            draw_lane_overlay(overlay, mask)
            draw_hud(overlay, lane_label, state, speed, gap, trend, width)

            # BEV mini-map inset
            bev_vis = cv2.resize(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), (260, 130))
            gray    = cv2.cvtColor(bev_vis, cv2.COLOR_BGR2GRAY)
            gray    = cv2.equalizeHist(gray)
            gray    = cv2.createCLAHE(clipLimit=3.0,
                                       tileGridSize=(4, 4)).apply(gray)
            bev_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            bev_vis = cv2.copyMakeBorder(bev_vis, 6, 6, 6, 6,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
            overlay[8:8 + bev_vis.shape[0],
                    -280:-280 + bev_vis.shape[1]] = bev_vis

            cv2.imshow("CARLA + LIME", overlay)

        if cv2.waitKey(1) == 27:
            break
finally:
    camera.stop()
    if vehicle is not None:
        vehicle.destroy()
    cv2.destroyAllWindows()
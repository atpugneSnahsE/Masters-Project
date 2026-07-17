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
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
import os

# --- PyTorch Grad-CAM Utilities ---
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

os.makedirs("xai_output", exist_ok=True)

# ====================== MODEL ======================
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=9,
    decoder_attention_type="scse",
)
checkpoint = torch.load(
    "/home/vgtu/Masters-Project/models/carla_lane_models/lane_model_best.pth",
    map_location="cpu",
    weights_only=False,
)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

transform = T.Compose([T.ToPILImage(), T.Resize((256, 512)), T.ToTensor()])

# ====================== GRAD-CAM CONFIGURATION ======================
# Target the deepest convolutional layer of your ResNet-34 encoder block
target_layers = [model.encoder.layer4[-1]]
cam_engine = GradCAM(model=model, target_layers=target_layers)

# ====================== XAI HELPERS ======================

def make_corridor_confidence(prob_vis, line_mask, h, w):
    """Generates confidence map smoothed around predicted model mask zones."""
    conf = prob_vis.copy()
    line_pixels = prob_vis[line_mask > 0]
    line_conf   = float(np.mean(line_pixels)) if len(line_pixels) > 0 else 0.85

    # Target lane mask regions get full prediction confidence
    conf[line_mask > 0] = prob_vis[line_mask > 0]
    conf = cv2.GaussianBlur(conf, (21, 21), 0)
    conf[:int(h * 0.35), :] = 0
    return conf


# ====================== GRAD-CAM GENERATION ======================
def generate_xai_figure(rgb_image, frame_id):
    save_path = f"xai_output/xai_frame_{frame_id:06d}.png"
    h, w = rgb_image.shape[:2]

    inp = transform(rgb_image).unsqueeze(0).to(device)
    
    # 1. Forward inference to capture output maps
    model.eval()
    with torch.no_grad():
        pred = model(inp)
    prob = torch.sigmoid(pred).cpu().numpy()
    prob = prepare_prob_map(prob)
    prob_vis = cv2.resize(prob, (w, h), cv2.INTER_LINEAR)
    prob_vis[:int(h * 0.38), :] = 0

    lower_half = prob_vis[int(h * 0.4):, :]
    thresh     = max(0.22, np.percentile(lower_half, 78))

    # Real deal: Raw model mask prediction 
    line_mask = (prob_vis > thresh).astype(np.uint8) * 255
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((11, 9), np.uint8))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN,  np.ones((7,  7), np.uint8))

    print(f"   Frame {frame_id} | Thresh: {thresh:.3f} | Mask px: {(line_mask > 0).sum()}")

    # ── FIXED BLENDED OVERLAY (No more phantom corridors) ──────────────────
    mask_overlay = rgb_image.copy()
    mask_overlay[line_mask > 0] = [0, 255, 120]  # Pure neon green 
    # Clean alpha blend directly matching what the network isolates
    enhanced = cv2.addWeighted(mask_overlay, 0.40, rgb_image, 0.60, 0)

    # ── GRAD-CAM INTERPOLATION ──────────────────────────────────────────────────
    cam_mask = cv2.resize(line_mask, (512, 256), interpolation=cv2.INTER_NEAREST) / 255.0
    targets = [SemanticSegmentationTarget(category=1, mask=cam_mask)]
    
    grayscale_cam = cam_engine(input_tensor=inp, targets=targets)[0, :]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h))
    gradcam_vis = show_cam_on_image(rgb_image.astype(np.float32) / 255.0, grayscale_cam_resized, use_rgb=True)

    # ── CONFIDENCE MAP ────────────────────────────────────────────────────────
    trust_map = make_corridor_confidence(prob_vis, line_mask, h, w)

    # ── PLOT ──────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    axs[0].imshow(rgb_image)
    axs[0].set_title("Input RGB")
    axs[0].axis('off')

    axs[1].imshow(enhanced)
    axs[1].set_title("True Model Detection")
    axs[1].axis('off')

    axs[2].imshow(gradcam_vis)
    axs[2].set_title("Grad-CAM Spatial Attribution")
    axs[2].axis('off')

    im = axs[3].imshow(trust_map, cmap='RdYlGn', vmin=0, vmax=1)
    axs[3].set_title("Lane Confidence Map")
    axs[3].axis('off')
    plt.colorbar(im, ax=axs[3])

    plt.suptitle(f"XAI Analysis (Grad-CAM) - Frame {frame_id}", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Grad-CAM Frame saved: {save_path}")


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


def prepare_prob_map(prob):
    prob = np.asarray(prob)
    if prob.ndim == 4:
        prob = np.squeeze(prob, axis=0)
    if prob.ndim == 3:
        if prob.shape[0] > 1 and prob.shape[0] != prob.shape[-1]:
            prob = np.max(prob[1:], axis=0)
        else:
            prob = prob[0]
    return np.squeeze(prob)


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
XAI_EVERY   = 1  # Real-time Grad-CAM allows processing frame-by-frame


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

    prob = torch.sigmoid(pred).cpu().numpy()
    prob = prepare_prob_map(prob)

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
show_window = True
try:
    while True:
        if latest is not None:
            rgb, mask, b, lane_label, gap, trend, state, width, speed = latest
            overlay = rgb.copy()

            overlay[mask == 255] = [0, 255, 120]
            draw_lane_overlay(overlay, mask)
            draw_hud(overlay, lane_label, state, speed, gap, trend, width)

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

            if show_window:
                try:
                    cv2.imshow("CARLA + GRAD-CAM", overlay)
                except cv2.error:
                    show_window = False
                    print("Headless environment detected; skipping OpenCV window display.")

        if show_window:
            try:
                if cv2.waitKey(1) == 27:
                    break
            except cv2.error:
                show_window = False
                print("Headless environment detected; skipping OpenCV keyboard handling.")
        else:
            import time
            time.sleep(0.1)
finally:
    camera.stop()
    if vehicle is not None:
        vehicle.destroy()
    if show_window:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
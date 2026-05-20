import timm.models.regnet
if not hasattr(timm.models.regnet, "RegNetCfg"):
    class RegNetCfg:
        pass
    timm.models.regnet.RegNetCfg = RegNetCfg

import carla
import torch
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from collections import deque
from enum import Enum

# ---------------- MODEL ----------------
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

# ---------------- CARLA SETUP ----------------
client = carla.Client("localhost", 2000)
client.set_timeout(10)

# Load Town10HD
world = client.load_world("Town04")

bp = world.get_blueprint_library()

# Use Volkswagen Amarok
vehicle_bp = bp.find('vehicle.jeep.wrangler_rubicon')

vehicle = None
for sp in world.get_map().get_spawn_points():
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle:
        break

if vehicle is None:
    print("Failed to spawn vehicle!")
    exit()

vehicle.set_autopilot(True)

# ================= NIGHT SETUP =================
def set_night_weather(world):
    weather = carla.WeatherParameters(
        cloudiness=85.0,
        precipitation=0.0,
        sun_altitude_angle=-90.0,
        sun_azimuth_angle=0.0,
        fog_density=6.0,
        fog_distance=12.0,
        wetness=15.0,
        fog_falloff=0.6
    )
    world.set_weather(weather)

def enable_night_lights(vehicle):
    # LowBeam + Position lights only (realistic night driving)
    lights = carla.VehicleLightState.HighBeam | carla.VehicleLightState.Position
    vehicle.set_light_state(carla.VehicleLightState(lights))

set_night_weather(world)
enable_night_lights(vehicle)

# Camera Setup
cam_bp = bp.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1024")
cam_bp.set_attribute("image_size_y", "512")

camera = world.spawn_actor(
    cam_bp,
    carla.Transform(carla.Location(x=1.35, z=1.25), carla.Rotation(pitch=-5)),
    attach_to=vehicle
)

latest = None

# ---------------- STATE ----------------
class LaneState(Enum):
    NORMAL = "NORMAL"
    CROSSWALK = "CROSSWALK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MERGING = "MERGING"
    CURVE = "CURVE"

left_hist = deque(maxlen=16)
right_hist = deque(maxlen=16)
_left_committed = None
_right_committed = None
_left_contrary = 0
_right_contrary = 0

mask_buf = deque(maxlen=5)
_lfit_buf = deque(maxlen=12)
_rfit_buf = deque(maxlen=12)

trend_buf = deque(maxlen=30)
trend_frame = 0

last_gap = None
last_trend = None
current_state = LaneState.NORMAL
freeze_count = 0
prev_left = "solid"
prev_right = "solid"

HYSTERESIS_N = 14

class KalmanFilter:
    def __init__(self, process_var=0.1, measurement_var=0.8):
        self.x = 0.0; self.p = 1.0; self.q = process_var; self.r = measurement_var
    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p = (1 - k) * self.p
        return self.x

kf_gap = KalmanFilter(0.08, 1.2)

# ---------------- NIGHT PREPROCESSING ----------------
def apply_night_preprocessing(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    
    # Gamma correction
    gamma = 0.55
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(enhanced, table)
    
    return enhanced

def carla_to_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1]

def get_speed_kmh():
    vel = vehicle.get_velocity()
    return (vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6

# ---------------- FILTERING & BEV ----------------
def filter_horizontal_noise(mask):
    clean = mask.copy()
    h, w = mask.shape
    for y in range(h):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0: continue
        span = xs[-1] - xs[0]
        if span > w * 0.60 or len(xs) < 8:
            clean[y] = 0
    return clean

def detect_crosswalk(mask):
    h, w = mask.shape
    roi = mask[int(h * 0.55):int(h * 0.85), :]
    rows = np.sum(roi > 0, axis=1)
    return bool(np.sum(rows > w * 0.35) > 12)

def bev(mask):
    h, w = mask.shape
    src = np.float32([[w*0.22, h*0.88], [w*0.78, h*0.88],
                      [w*0.38, h*0.63], [w*0.62, h*0.63]])
    dst = np.float32([[w*0.25, h], [w*0.75, h],
                      [w*0.25, 0], [w*0.75, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(mask, M, (w, h))

# ---------------- CLASSIFICATION (Strict Double) ----------------
def get_line_type(band):
    row_hits = np.sum(band > 0, axis=1) > 2
    occ = np.mean(row_hits)
    return "solid" if occ > 0.68 else "dashed" if occ > 0.16 else "unknown"

def classify_boundary(side, hist, side_id):
    global _left_contrary, _right_contrary, _left_committed, _right_committed
    h, w = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col = int(np.argmax(col_profile))
    peak_val = int(col_profile[peak_col])

    if peak_val < 5:
        raw, conf = "unknown", 0.25
    else:
        search = col_profile.astype(float).copy()
        suppress = max(4, int(w * 0.055))
        search[max(0, peak_col-suppress):min(w, peak_col+suppress)] = 0
        second_col = int(np.argmax(search))
        second_val = int(search[second_col])

        distance = abs(second_col - peak_col)
        is_double_candidate = (second_val >= peak_val * 0.55 and 8 <= distance <= int(w * 0.105))

        bh = max(6, int(w * 0.11))
        main_band = side[:, max(0, peak_col-bh):min(w, peak_col+bh)]
        main_type = get_line_type(main_band)

        if is_double_candidate:
            second_band = side[:, max(0, second_col-bh):min(w, second_col+bh)]
            second_type = get_line_type(second_band)
            raw = f"double {main_type}" if main_type == second_type and main_type != "unknown" else main_type
        else:
            raw = main_type

        conf = min(1.0, peak_val / 30.0)
        conf *= 0.78 if "solid" in raw else 0.92
        if "double" in raw: conf *= 0.88

    hist.append((raw, conf))
    vote = _weighted_majority(hist) if len(hist) >= 5 else raw

    if side_id == 'left':
        committed, contrary = _left_committed, _left_contrary
    else:
        committed, contrary = _right_committed, _right_contrary

    if committed is None:
        committed = vote
        contrary = 0
    elif vote == committed:
        contrary = 0
    else:
        contrary += 1
        if contrary >= HYSTERESIS_N:
            committed = vote
            contrary = 0

    if side_id == 'left':
        _left_committed, _left_contrary = committed, contrary
    else:
        _right_committed, _right_contrary = committed, contrary

    return committed

def _weighted_majority(hist):
    real = [x for x, c in hist if x != "unknown"]
    pool = real if len(real) >= 3 else [x for x, c in hist]
    if not pool: return "unknown"
    from collections import defaultdict
    scores = defaultdict(float)
    for label, conf in hist:
        if label != "unknown" or len(real) < 3:
            scores[label] += conf
    return max(scores, key=scores.get)

# ---------------- REMAINING FUNCTIONS ----------------
def compute_lane_width(mask):
    h, w = mask.shape
    mid = w // 2
    widths = []
    for y in range(int(h * 0.6), h, 4):
        row = np.where(mask[y] > 0)[0]
        if len(row) == 0: continue
        L = row[row < mid]
        R = row[row >= mid]
        if len(L) > 3 and len(R) > 3:
            widths.append(np.min(R) - np.max(L))
    return np.median(widths) if widths else 0

def measure_dash_trend(side, speed_kmh):
    global trend_frame
    trend_frame += 1
    h, w = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col = int(np.argmax(col_profile))
    if col_profile[peak_col] < 5:
        return None, None

    bh = max(6, int(w * 0.13))
    band = side[:, max(0, peak_col-bh):min(w, peak_col+bh)]
    signal = (np.sum(band > 0, axis=1) > 2).astype(np.int8)
    diff = np.diff(signal)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return None, None
    if len(ends) > 0 and len(starts) > 0 and ends[0] < starts[0]:
        ends = ends[1:]

    n = min(len(starts), len(ends))
    if n < 2: return None, None

    gaps = [int(starts[i+1]) - int(ends[i]) for i in range(n-1)
            if 5 < (starts[i+1] - ends[i]) < h * 0.6]

    if not gaps: return None, None

    mean_gap = float(np.mean(gaps))
    smoothed = kf_gap.update(mean_gap)
    trend_buf.append((trend_frame, smoothed))

    if len(trend_buf) < 12:
        return smoothed, "calculating..."

    recent = list(trend_buf)[-15:]
    data = np.array(recent)
    slope = np.polyfit(data[:, 0], data[:, 1], 1)[0]

    trend = "diverging" if slope > 0.20 else "converging" if slope < -0.20 else "constant"
    return smoothed, trend

def update_state(lt, rt, gap, crosswalk, lane_width):
    global current_state, freeze_count
    if crosswalk:
        current_state = LaneState.CROSSWALK
        freeze_count = 12
        return
    if lane_width < 70 or lane_width > 480:
        current_state = LaneState.LOW_CONFIDENCE
        freeze_count = max(freeze_count, 6)
        return
    if gap is not None and gap > 180 and ("dashed" in lt or "dashed" in rt):
        current_state = LaneState.MERGING
    else:
        current_state = LaneState.NORMAL

def make_lane_label(lt, rt):
    if lt == rt == "unknown": return "UNKNOWN"
    if lt == "solid" and rt == "solid": return "DOUBLE SOLID"
    if lt == "dashed" and rt == "dashed": return "DOUBLE DASHED"
    return f"{lt} | {rt}"

# ---------------- DRAW FUNCTIONS ----------------
def draw_lane_overlay(overlay, mask):
    h, w = mask.shape
    mid = w // 2
    l_pts, r_pts = [], []

    for y in range(int(h * 0.62), h, 3):
        row = np.where(mask[y] > 0)[0]
        if len(row) < 8: continue
        L = row[row < mid]
        R = row[row >= mid]
        if len(L) > 5: l_pts.append([float(y), float(np.max(L))])
        if len(R) > 5: r_pts.append([float(y), float(np.min(R))])

    if len(l_pts) >= 8:
        lp = np.array(l_pts)
        try: _lfit_buf.append(np.polyfit(lp[:,0], lp[:,1], 2))
        except: pass
    if len(r_pts) >= 8:
        rp = np.array(r_pts)
        try: _rfit_buf.append(np.polyfit(rp[:,0], rp[:,1], 2))
        except: pass

    if len(_lfit_buf) < 3 or len(_rfit_buf) < 3:
        return

    lfit = np.mean(list(_lfit_buf)[-8:], axis=0)
    rfit = np.mean(list(_rfit_buf)[-8:], axis=0)

    ys = np.arange(int(h * 0.62), h, dtype=np.float32)
    lxs = np.polyval(lfit, ys).clip(0, mid-1).astype(np.int32)
    rxs = np.polyval(rfit, ys).clip(mid, w-1).astype(np.int32)
    yi = ys.astype(np.int32)

    fill = overlay.copy()
    pts = np.concatenate([np.stack([lxs, yi], axis=1),
                          np.stack([rxs, yi], axis=1)[::-1]], axis=0)
    cv2.fillPoly(fill, [pts], (0, 180, 0))
    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)

    for i in range(len(yi)-1):
        cv2.line(overlay, (lxs[i], yi[i]), (lxs[i+1], yi[i+1]), (0, 255, 255), 4)
        cv2.line(overlay, (rxs[i], yi[i]), (rxs[i+1], yi[i+1]), (0, 255, 255), 4)

def draw_hud(overlay, lane_label, state, speed, gap, trend, width):
    h, w = overlay.shape[:2]
    bar = np.zeros((145, w, 3), dtype=np.uint8)
    cv2.rectangle(bar, (0,0), (w, 145), (0, 0, 0), -1)
    overlay[0:145] = cv2.addWeighted(overlay[0:145], 0.35, bar, 0.65, 0)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(overlay, f"Lane: {lane_label}", (30, 45), font, 1.05, (0, 255, 255), 2)
    cv2.putText(overlay, f"State: {state.value} | Speed: {speed:.0f} km/h", (30, 80), font, 0.85, (0, 255, 180), 2)

    info = f"Gap: {gap:.1f}px | Trend: {trend} | Width: {int(width)}px" if gap is not None else f"Width: {int(width)}px"
    cv2.putText(overlay, info, (30, 120), font, 0.82, (255, 255, 100), 2)

    gx = w - 380
    gy = h - 60
    cv2.putText(overlay, "L", (gx-30, gy+22), font, 0.85, (200,200,200), 2)
    cv2.putText(overlay, "R", (gx+220, gy+22), font, 0.85, (200,200,200), 2)
    cv2.rectangle(overlay, (gx, gy), (gx+200, gy+28), (40,40,40), -1)
    cv2.rectangle(overlay, (gx, gy), (gx+200, gy+28), (255,255,255), 2)
    cv2.line(overlay, (gx+98, gy), (gx+102, gy+28), (80,80,80), 3)

    offset = 0
    if width > 100:
        offset = (width - 300) / 7
    pos = int(gx + 100 + np.clip(offset, -90, 90))
    cv2.rectangle(overlay, (pos-10, gy-3), (pos+10, gy+31), (0, 180, 255), -1)

# ---------------- CALLBACK ----------------
def process(image):
    global latest, freeze_count, prev_left, prev_right, last_gap, last_trend, current_state

    raw_rgb = carla_to_rgb(image)
    rgb = apply_night_preprocessing(raw_rgb)
    speed_kmh = get_speed_kmh()

    inp = transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)

    prob = torch.sigmoid(pred).cpu().squeeze().numpy()
    lower_half = prob[int(prob.shape[0]*0.4):, :]
    thresh = max(0.32, np.percentile(lower_half, 87))
    mask = (prob > thresh).astype(np.uint8) * 255

    if speed_kmh > 60:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 5), np.uint8))
    else:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask = cv2.resize(mask, (raw_rgb.shape[1], raw_rgb.shape[0]), cv2.INTER_NEAREST)
    mask = filter_horizontal_noise(mask)

    mask_buf.append(mask)
    fused = np.median(np.array(mask_buf), axis=0).astype(np.uint8)

    crosswalk = detect_crosswalk(fused)
    b = bev(fused)
    left = b[:, :b.shape[1]//2]
    right = b[:, b.shape[1]//2:]

    if freeze_count > 0:
        lt, rt = prev_left, prev_right
        freeze_count -= 1
    else:
        lt = classify_boundary(left, left_hist, 'left')
        rt = classify_boundary(right, right_hist, 'right')
        prev_left, prev_right = lt, rt

    lane_label = make_lane_label(lt, rt)
    lane_width = compute_lane_width(fused)

    side = left if "dashed" in lt else right if "dashed" in rt else None
    gap = trend = None
    if side is not None and not crosswalk:
        gap, trend = measure_dash_trend(side, speed_kmh)
        if gap is not None:
            last_gap, last_trend = gap, trend

    update_state(lt, rt, gap, crosswalk, lane_width)

    latest = (raw_rgb, fused, b, lane_label, last_gap, last_trend, current_state, lane_width, speed_kmh)


camera.listen(process)

# ---------------- MAIN ----------------
try:
    while True:
        if latest is not None:
            rgb, mask, b, lane_label, gap, trend, state, width, speed = latest
            overlay = rgb.copy()

            overlay[mask == 255] = [0, 255, 120]
            draw_lane_overlay(overlay, mask)
            draw_hud(overlay, lane_label, state, speed, gap, trend, width)

            # Enhanced Night BEV
            bev_vis = cv2.resize(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), (260, 130))
            gray = cv2.cvtColor(bev_vis, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4)).apply(gray)
            bev_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            bev_vis = cv2.copyMakeBorder(bev_vis, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255,255,255))
            overlay[8:8+bev_vis.shape[0], -280:-280+bev_vis.shape[1]] = bev_vis

            cv2.imshow("EU Lane Understanding - NIGHT (Town10HD)", overlay)

        if cv2.waitKey(1) == 27:
            break

finally:
    camera.stop()
    try:
        if vehicle is not None:
            vehicle.destroy()
    except: pass
    cv2.destroyAllWindows()

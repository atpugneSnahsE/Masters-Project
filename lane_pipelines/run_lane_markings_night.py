import carla
import torch
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
import time
import os
import threading
import queue
import warnings

warnings.filterwarnings("ignore")

os.makedirs("xai_output", exist_ok=True)

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================
NUM_CLASSES = 9
EGO_CLASS   = 1

# ── Step 1: load checkpoint first so we can inspect its keys ──
state_dict = torch.load("models/lane_model_best.pth", map_location="cpu")
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

# ── Step 2: auto-detect whether the saved weights include scSE attention ──
has_attention  = any("attention" in k for k in state_dict.keys())
attention_type = "scse" if has_attention else None
print(f"ℹ️  Checkpoint attention type: {'scse' if has_attention else 'None (plain decoder)'}")

# ── Step 3: build the model that matches the checkpoint ──
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_attention_type=attention_type,
    activation=None
)

model.load_state_dict(state_dict, strict=True)

device  = "cuda" if torch.cuda.is_available() else "cpu"
model   = model.to(device)
model.eval()

# Half-precision for real-time acceleration on CUDA
is_half = (device == "cuda")
if is_half:
    model = model.half()
    torch.backends.cudnn.benchmark = True

print(f"✅ Loaded model with {model.segmentation_head[0].out_channels} classes on {device.upper()}")

_INF_H, _INF_W = 256, 512
_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

if is_half:
    _MEAN = _MEAN.half()
    _STD  = _STD.half()


def preprocess_gpu(rgb_uint8):
    """
    Transforms raw CARLA images to properly shaped tensors for inference.
    """
    small = cv2.resize(rgb_uint8, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(small).permute(2, 0, 1).to(device)
    if is_half:
        t = t.half()
    else:
        t = t.float()
    t = t.div_(255.0).unsqueeze(0)
    t = (t - _MEAN) / _STD
    return t


# ==========================================================
# LIME XAI METRICS
# ==========================================================
_xai_running = False
_xai_lock    = threading.Lock()


def batch_predict(images):
    """
    Fixed interface to accurately bridge LIME's internal image
    generation arrays with PyTorch's forward channel format.
    """
    processed = []
    for img in images:
        # Scale back to standard unit integers if LIME feeds normalised arrays
        if img.dtype != np.uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

        img_resized = cv2.resize(img, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)
        processed.append(img_resized)

    # Stack into NumPy array matrix -> shape (N, H, W, C)
    batch_np = np.stack(processed)
    # Re-order axes to match PyTorch expectations: shape (N, C, H, W)
    batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(device)

    if is_half:
        batch_tensor = batch_tensor.half()
    else:
        batch_tensor = batch_tensor.float()

    batch_tensor = batch_tensor.div_(255.0)

    # Apply standard ImageNet normalisations
    batch_tensor = (batch_tensor - _MEAN) / _STD

    with torch.no_grad():
        preds = model(batch_tensor)          # (N, NUM_CLASSES, H, W)
        probs = torch.softmax(preds.float(), dim=1)

    # Average across spatial dimensions to get per-class probabilities
    probs_per_class = probs.mean(dim=(2, 3)).cpu().numpy()
    return probs_per_class.astype(np.float64)


def _xai_worker(rgb_image, frame_id):
    """
    Worker pipeline handling regional image slicing and visual generation.
    """
    global _xai_running
    with _xai_lock:
        if _xai_running:
            return
        _xai_running = True

    try:
        save_path = f"xai_output/xai_frame_{frame_id:06d}.png"
        h, w      = rgb_image.shape[:2]

        inp = preprocess_gpu(rgb_image)
        with torch.no_grad():
            pred = model(inp)

        prob_all = torch.softmax(pred.float(), dim=1).cpu().squeeze(0)
        prob_ego = prob_all[EGO_CLASS].numpy()

        prob_vis = cv2.resize(prob_ego, (w, h), interpolation=cv2.INTER_LINEAR)

        # ── Region Extraction ──
        roi        = prob_vis[int(h * 0.4):, :]
        base_thresh = max(0.08, np.percentile(roi, 75))
        binary     = (prob_vis > base_thresh).astype(np.uint8) * 255
        binary     = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        binary     = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if n_labels > 1:
            ego_lane = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
        else:
            ego_lane = binary

        # ── LIME ──
        explainer = lime_image.LimeImageExplainer(verbose=False)
        segmenter = lambda x: quickshift(x, kernel_size=4, max_dist=20, ratio=0.2)

        explanation = explainer.explain_instance(
            rgb_image.astype(np.float64) / 255.0,
            batch_predict,
            top_labels=1,
            hide_color=0,
            num_features=25,
            num_samples=150,
            batch_size=16,
            segmentation_fn=segmenter,
            random_seed=42
        )

        temp, lime_mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=12,
            hide_rest=False
        )

        trust_map = cv2.GaussianBlur(prob_vis, (21, 21), 0)
        trust_map = (trust_map - trust_map.min()) / (trust_map.max() - trust_map.min() + 1e-8)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(rgb_image);               axs[0].set_title("Input RGB");          axs[0].axis('off')
        axs[1].imshow(ego_lane, cmap='gray');   axs[1].set_title("Detected Lane Area"); axs[1].axis('off')
        axs[2].imshow(mark_boundaries(temp, lime_mask)); axs[2].set_title("LIME Attribution"); axs[2].axis('off')
        im = axs[3].imshow(trust_map, cmap='RdYlGn', vmin=0, vmax=1)
        axs[3].set_title("Trust Score Map"); axs[3].axis('off')
        plt.colorbar(im, ax=axs[3])
        plt.suptitle(f"XAI Analysis - Frame {frame_id} | Avg Conf: {prob_vis.mean():.4f}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        plt.close()
        print(f"✅ XAI saved: {save_path}")

    except Exception as e:
        import traceback
        print(f"XAI Error: {e}")
        traceback.print_exc()
    finally:
        with _xai_lock:
            _xai_running = False


def maybe_launch_xai(raw_rgb, frame_id):
    if _xai_running:
        return
    threading.Thread(target=_xai_worker, args=(raw_rgb.copy(), frame_id), daemon=True).start()


# ==========================================================
# TRAFFIC CONFIGURATION  (edit these to taste)
# ==========================================================
NUM_NPC_VEHICLES  = 40   # number of NPC cars/trucks to spawn
NUM_NPC_WALKERS   = 20   # number of pedestrians to spawn

# ==========================================================
# CARLA ENVIRONMENT SETUP
# ==========================================================
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

print("Loading Town...")
world = client.load_world("Town10HD")

settings = world.get_settings()
settings.synchronous_mode    = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

weather = carla.WeatherParameters(
    cloudiness=20.0, precipitation=0.0, fog_density=0.0,
    wetness=0.0, sun_altitude_angle=-35.0
)
world.set_weather(weather)

blueprints       = world.get_blueprint_library()
vehicle_bp       = blueprints.find("vehicle.tesla.model3")
spawn_points     = world.get_map().get_spawn_points()
spawn_transform  = spawn_points[20]

vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
if vehicle is None:
    raise RuntimeError("Vehicle spawn failed")
print("Vehicle spawned")

vehicle.set_autopilot(True)
traffic_manager.vehicle_percentage_speed_difference(vehicle, 10)
world.tick()

_headlight_state = carla.VehicleLightState(
    carla.VehicleLightState.Position |
    carla.VehicleLightState.LowBeam  |
    carla.VehicleLightState.HighBeam |
    carla.VehicleLightState.Fog
)
vehicle.set_light_state(_headlight_state)
print("Headlights ON")

camera_bp = blueprints.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "640")
camera_bp.set_attribute("image_size_y", "360")
camera_bp.set_attribute("fov",          "100")

camera_transform = carla.Transform(
    carla.Location(x=0.25, y=0.0, z=1.45),
    carla.Rotation(pitch=-5)
)
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)


# ==========================================================
# TRAFFIC SPAWNING  (NPC vehicles + pedestrians)
# ==========================================================
npc_vehicles  = []   # list of spawned NPC vehicle actors
npc_walkers   = []   # list of spawned walker actors
npc_walker_ai = []   # list of spawned walker AI controller actors

# ── NPC Vehicles ──────────────────────────────────────────
vehicle_filter = [
    "vehicle.audi.*", "vehicle.bmw.*", "vehicle.chevrolet.*",
    "vehicle.citroen.*", "vehicle.dodge.*", "vehicle.ford.*",
    "vehicle.jeep.*", "vehicle.lincoln.*", "vehicle.mercedes.*",
    "vehicle.mini.*", "vehicle.mustang.*", "vehicle.nissan.*",
    "vehicle.seat.*", "vehicle.tesla.*", "vehicle.toyota.*",
    "vehicle.volkswagen.*", "vehicle.volvo.*",
]

npc_bps = []
for filt in vehicle_filter:
    npc_bps.extend(blueprints.filter(filt))

# Exclude bikes/motorcycles (2-wheelers behave erratically at night)
npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]

# Use spawn points that are NOT the ego vehicle's point
ego_sp_idx     = 20
available_sps  = [sp for i, sp in enumerate(spawn_points) if i != ego_sp_idx]

import random
random.shuffle(available_sps)
spawn_cmds = []
used_bps   = []
for i in range(min(NUM_NPC_VEHICLES, len(available_sps))):
    bp = random.choice(npc_bps)
    # Randomise colour if the blueprint supports it
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    spawn_cmds.append(carla.command.SpawnActor(bp, available_sps[i])
                      .then(carla.command.SetAutopilot(carla.command.FutureActor, True,
                                                       traffic_manager.get_port())))
    used_bps.append(bp)

results = client.apply_batch_sync(spawn_cmds, True)
for res in results:
    if not res.error:
        actor = world.get_actor(res.actor_id)
        if actor:
            # Night lights on every NPC
            actor.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position |
                carla.VehicleLightState.LowBeam
            ))
            # Slightly varied speeds so traffic feels organic
            traffic_manager.vehicle_percentage_speed_difference(
                actor, random.uniform(-15, 20))
            traffic_manager.distance_to_leading_vehicle(actor, random.uniform(1.5, 4.0))
            npc_vehicles.append(actor)
    else:
        pass  # spawn point was occupied — skip silently

print(f"🚗 Spawned {len(npc_vehicles)} NPC vehicles")

# ── Pedestrians ───────────────────────────────────────────
walker_bps = blueprints.filter("walker.pedestrian.*")

walker_spawn_cmds = []
walker_spawn_locs = []
for _ in range(NUM_NPC_WALKERS):
    loc = world.get_random_location_from_navigation()
    if loc is None:
        continue
    sp  = carla.Transform(loc)
    bp  = random.choice(walker_bps)
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "false")
    walker_spawn_cmds.append(carla.command.SpawnActor(bp, sp))
    walker_spawn_locs.append(sp)

walker_results = client.apply_batch_sync(walker_spawn_cmds, True)
walker_ids     = [r.actor_id for r in walker_results if not r.error]
walker_actors  = world.get_actors(walker_ids)
npc_walkers.extend(walker_actors)

# Spawn AI controllers for each successfully created walker
ai_bp          = blueprints.find("controller.ai.walker")
ai_spawn_cmds  = [carla.command.SpawnActor(ai_bp, carla.Transform(), w)
                  for w in walker_actors]
ai_results     = client.apply_batch_sync(ai_spawn_cmds, True)
ai_ids         = [r.actor_id for r in ai_results if not r.error]
ai_actors      = world.get_actors(ai_ids)
npc_walker_ai.extend(ai_actors)

world.tick()   # one tick so controllers are properly registered

# Start each AI controller walking to a random destination
for ctrl in ai_actors:
    ctrl.start()
    ctrl.go_to_location(world.get_random_location_from_navigation())
    ctrl.set_max_speed(random.uniform(1.0, 2.2))

print(f"🚶 Spawned {len(npc_walkers)} pedestrians")
world.tick()


# ==========================================================
# STATE & HELPERS
# ==========================================================
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

mask_buf   = deque(maxlen=5)
_lfit_buf  = deque(maxlen=12)
_rfit_buf  = deque(maxlen=12)
trend_buf  = deque(maxlen=30)
trend_frame = 0

last_gap      = None
last_trend    = None
current_state = LaneState.NORMAL
freeze_count  = 0
prev_left     = "solid"
prev_right    = "solid"

frame_queue   = queue.Queue(maxsize=4)
frame_count   = 0
shutdown_flag = False

HYSTERESIS_N = 14
XAI_EVERY    = 60

kf_gap_x = 0.0
kf_gap_p = 1.0
KF_Q, KF_R = 0.08, 1.2


def kf_update(z):
    global kf_gap_x, kf_gap_p
    kf_gap_p += KF_Q
    k         = kf_gap_p / (kf_gap_p + KF_R)
    kf_gap_x  = kf_gap_x + k * (z - kf_gap_x)
    kf_gap_p  = (1 - k) * kf_gap_p
    return kf_gap_x


_clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))


def night_enhance(rgb):
    lab       = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b   = cv2.split(lab)
    l         = _clahe.apply(l)
    rgb_out   = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    inv_gamma = 1.0 / 0.72
    lut       = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    rgb_out   = cv2.LUT(rgb_out, lut)
    rgb_out   = cv2.convertScaleAbs(rgb_out, alpha=1.35, beta=8)
    return rgb_out


def carla_to_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()


def get_speed_kmh():
    v = vehicle.get_velocity()
    return (v.x**2 + v.y**2 + v.z**2) ** 0.5 * 3.6


def filter_horizontal_noise(mask):
    clean = mask.copy()
    h, w  = mask.shape
    for y in range(h):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        if xs[-1] - xs[0] > w * 0.58 or len(xs) < 6:
            clean[y] = 0
    return clean


def detect_crosswalk(mask):
    h, w = mask.shape
    roi  = mask[int(h * 0.55):int(h * 0.85), :]
    rows = np.sum(roi > 0, axis=1)
    return bool(np.sum(rows > w * 0.35) > 12)


def bev(mask):
    h, w = mask.shape
    src  = np.float32([[w*0.18, h*0.92], [w*0.82, h*0.92],
                       [w*0.42, h*0.58], [w*0.58, h*0.58]])
    dst  = np.float32([[w*0.20, h],      [w*0.80, h],
                       [w*0.20, 0],      [w*0.80, 0]])
    M    = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(mask, M, (w, h))


def get_line_type(band):
    occ = np.mean(np.sum(band > 0, axis=1) > 2)
    return "solid" if occ > 0.68 else "dashed" if occ > 0.16 else "unknown"


def classify_boundary(side, hist, side_id):
    global _left_committed, _right_committed, _left_contrary, _right_contrary
    h, w        = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col    = int(np.argmax(col_profile))
    peak_val    = int(col_profile[peak_col])

    if peak_val < 4:
        raw, conf = "unknown", 0.25
    else:
        search   = col_profile.astype(float).copy()
        suppress = max(4, int(w * 0.055))
        search[max(0, peak_col - suppress):min(w, peak_col + suppress)] = 0
        second_col = int(np.argmax(search))
        second_val = int(search[second_col])
        distance   = abs(second_col - peak_col)
        is_double  = (second_val >= peak_val * 0.55 and 8 <= distance <= int(w * 0.11))

        bh        = max(6, int(w * 0.11))
        main_band = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
        main_type = get_line_type(main_band)

        if is_double:
            sb_band     = side[:, max(0, second_col - bh):min(w, second_col + bh)]
            second_type = get_line_type(sb_band)
            raw = f"double {main_type}" if main_type == second_type != "unknown" else main_type
        else:
            raw = main_type

        conf  = min(1.0, peak_val / 25.0)
        conf *= 0.78 if "solid" in raw else 0.92
        if "double" in raw:
            conf *= 0.88

    hist.append((raw, conf))
    vote = _weighted_majority(hist) if len(hist) >= 5 else raw

    committed = _left_committed  if side_id == 'left' else _right_committed
    contrary  = _left_contrary   if side_id == 'left' else _right_contrary

    if committed is None:
        committed, contrary = vote, 0
    elif vote == committed:
        contrary = 0
    else:
        contrary += 1
        if contrary >= HYSTERESIS_N:
            committed, contrary = vote, 0

    if side_id == 'left':
        _left_committed,  _left_contrary  = committed, contrary
    else:
        _right_committed, _right_contrary = committed, contrary
    return committed


def _weighted_majority(hist):
    from collections import defaultdict
    real = [x for x, _ in hist if x != "unknown"]
    pool = real if len(real) >= 3 else [x for x, _ in hist]
    if not pool:
        return "unknown"
    scores = defaultdict(float)
    for label, conf in hist:
        if label != "unknown" or len(real) < 3:
            scores[label] += conf
    return max(scores, key=scores.get)


def compute_lane_width(mask):
    h, w, mid = mask.shape[0], mask.shape[1], mask.shape[1] // 2
    widths = []
    for y in range(int(h * 0.6), h, 4):
        row = np.where(mask[y] > 0)[0]
        if len(row) == 0:
            continue
        L, R = row[row < mid], row[row >= mid]
        if len(L) > 3 and len(R) > 3:
            widths.append(int(np.min(R)) - int(np.max(L)))
    return float(np.median(widths)) if widths else 0.0


def measure_dash_trend(side, speed_kmh):
    global trend_frame
    trend_frame += 1
    h, w        = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col    = int(np.argmax(col_profile))
    if col_profile[peak_col] < 4:
        return None, None

    bh     = max(6, int(w * 0.13))
    band   = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
    signal = (np.sum(band > 0, axis=1) > 2).astype(np.int8)
    diff   = np.diff(signal)
    starts = np.where(diff ==  1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return None, None
    if ends.size and ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    if n < 2:
        return None, None

    gaps = [int(starts[i + 1]) - int(ends[i]) for i in range(n - 1)
            if 5 < (starts[i + 1] - ends[i]) < h * 0.6]
    if not gaps:
        return None, None

    smoothed = kf_update(float(np.mean(gaps)))
    trend_buf.append((trend_frame, smoothed))
    if len(trend_buf) < 12:
        return smoothed, "calculating..."

    data  = np.array(list(trend_buf)[-15:])
    slope = np.polyfit(data[:, 0], data[:, 1], 1)[0]
    trend = "diverging" if slope > 0.20 else "converging" if slope < -0.20 else "constant"
    return smoothed, trend


def update_state(lt, rt, gap, crosswalk, lane_width):
    global current_state, freeze_count
    if crosswalk:
        current_state = LaneState.CROSSWALK
        freeze_count  = 12
        return
    if lane_width < 60 or lane_width > 500:
        current_state = LaneState.LOW_CONFIDENCE
        freeze_count  = max(freeze_count, 6)
        return
    current_state = (
        LaneState.MERGING
        if (gap is not None and gap > 180 and ("dashed" in lt or "dashed" in rt))
        else LaneState.NORMAL
    )


def make_lane_label(lt, rt):
    if lt == rt == "unknown":             return "UNKNOWN"
    if lt == "solid"  and rt == "solid":  return "DOUBLE SOLID"
    if lt == "dashed" and rt == "dashed": return "DOUBLE DASHED"
    return f"{lt} | {rt}"


def draw_lane_overlay(overlay, mask):
    h, w, mid = mask.shape[0], mask.shape[1], mask.shape[1] // 2
    l_pts, r_pts = [], []
    for y in range(int(h * 0.50), h, 3):
        row = np.where(mask[y] > 0)[0]
        if len(row) < 6:
            continue
        L, R = row[row < mid], row[row >= mid]
        if len(L) > 4: l_pts.append([float(y), float(np.max(L))])
        if len(R) > 4: r_pts.append([float(y), float(np.min(R))])

    if len(l_pts) >= 6:
        try:
            _lfit_buf.append(np.polyfit(np.array(l_pts)[:, 0], np.array(l_pts)[:, 1], 2))
        except Exception:
            pass
    if len(r_pts) >= 6:
        try:
            _rfit_buf.append(np.polyfit(np.array(r_pts)[:, 0], np.array(r_pts)[:, 1], 2))
        except Exception:
            pass

    if len(_lfit_buf) < 3 or len(_rfit_buf) < 3:
        return

    lfit = np.mean(list(_lfit_buf)[-8:], axis=0)
    rfit = np.mean(list(_rfit_buf)[-8:], axis=0)

    ys  = np.arange(int(h * 0.50), h, dtype=np.float32)
    lxs = np.polyval(lfit, ys).clip(0, mid - 1).astype(np.int32)
    rxs = np.polyval(rfit, ys).clip(mid, w - 1).astype(np.int32)
    yi  = ys.astype(np.int32)

    fill = overlay.copy()
    pts  = np.concatenate([np.stack([lxs, yi], axis=1),
                           np.stack([rxs, yi], axis=1)[::-1]], axis=0)
    cv2.fillPoly(fill, [pts], (0, 180, 0))
    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)
    for i in range(len(yi) - 1):
        cv2.line(overlay, (lxs[i], yi[i]), (lxs[i + 1], yi[i + 1]), (0, 255, 255), 3)
        cv2.line(overlay, (rxs[i], yi[i]), (rxs[i + 1], yi[i + 1]), (0, 255, 255), 3)


def draw_hud(overlay, lane_label, state, speed, gap, trend, width):
    h, w = overlay.shape[:2]
    bar  = np.zeros((130, w, 3), dtype=np.uint8)
    overlay[0:130] = cv2.addWeighted(overlay[0:130], 0.35, bar, 0.65, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(overlay, f"Lane: {lane_label}",
                (20, 38),  font, 0.85, (0, 255, 255), 2)
    cv2.putText(overlay, f"State: {state.value} | Speed: {speed:.0f} km/h",
                (20, 68),  font, 0.72, (0, 255, 180), 2)
    info = (f"Gap: {gap:.1f}px | Trend: {trend} | Width: {int(width)}px"
            if gap is not None else f"Width: {int(width)}px")
    cv2.putText(overlay, info, (20, 98), font, 0.68, (255, 255, 100), 2)
    # Traffic info (top-right corner)
    traffic_txt = f"NPC Cars: {len(npc_vehicles)}  Walkers: {len(npc_walkers)}"
    txt_size    = cv2.getTextSize(traffic_txt, font, 0.58, 1)[0]
    cv2.putText(overlay, traffic_txt,
                (w - txt_size[0] - 14, 28), font, 0.58, (200, 200, 255), 1)


# ==========================================================
# THREAD-SAFE CAMERA CALLBACK
# ==========================================================
def process_image(image):
    global frame_count, freeze_count, prev_left, prev_right, last_gap, last_trend, current_state

    if shutdown_flag:
        return

    frame_count += 1
    raw_rgb   = carla_to_rgb(image)
    rgb       = night_enhance(raw_rgb)
    speed_kmh = get_speed_kmh()

    inp = preprocess_gpu(rgb)
    with torch.no_grad():
        pred = model(inp)

    prob_all = torch.softmax(pred.float(), dim=1)
    avg_conf = float(prob_all[0, EGO_CLASS].mean().item())
    prob_ego_hw = prob_all[0, EGO_CLASS].cpu().numpy()

    prob_vis = cv2.resize(prob_ego_hw,
                          (raw_rgb.shape[1], raw_rgb.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    ego_thresh = 0.5
    mask = (prob_vis > ego_thresh).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = filter_horizontal_noise(mask)

    mask_buf.append(mask)
    fused = np.median(np.array(mask_buf), axis=0).astype(np.uint8)

    crosswalk  = detect_crosswalk(fused)
    b          = bev(fused)
    left_half  = b[:, :b.shape[1] // 2]
    right_half = b[:, b.shape[1] // 2:]

    if freeze_count > 0:
        lt, rt       = prev_left, prev_right
        freeze_count -= 1
    else:
        lt = classify_boundary(left_half,  left_hist,  'left')
        rt = classify_boundary(right_half, right_hist, 'right')
        prev_left, prev_right = lt, rt

    lane_label = make_lane_label(lt, rt)
    lane_width = compute_lane_width(fused)

    side = left_half if "dashed" in lt else right_half if "dashed" in rt else None
    gap = trend = None
    if side is not None and not crosswalk:
        gap, trend = measure_dash_trend(side, speed_kmh)
        if gap is not None:
            last_gap, last_trend = gap, trend

    update_state(lt, rt, gap, crosswalk, lane_width)

    if frame_count % XAI_EVERY == 0:
        maybe_launch_xai(raw_rgb, frame_count)

    frame_data = (raw_rgb, fused, b, lane_label, last_gap, last_trend,
                  current_state, lane_width, speed_kmh, avg_conf, prob_vis)
    try:
        frame_queue.put_nowait(frame_data)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait(frame_data)


# ==========================================================
# STREAMING INITIALISATION
# ==========================================================
camera.listen(process_image)

print("Warming up...")
for i in range(80):
    world.tick()
    if i % 10 == 0:
        vehicle.set_light_state(_headlight_state)
time.sleep(1)
print("Headlights confirmed ON")
print("🚀 CARLA Night Lane XAI Online | ESC to exit")

# ==========================================================
# MAIN LOOP
# ==========================================================
try:
    while True:
        world.tick()

        latest_frame = None
        while not frame_queue.empty():
            try:
                latest_frame = frame_queue.get_nowait()
            except queue.Empty:
                break

        if latest_frame is not None:
            raw_rgb, mask, b, lane_label, gap, trend, state, width, speed, avg_conf, prob_vis = latest_frame

            overlay = raw_rgb.copy()
            overlay[mask == 255] = [0, 255, 120]
            draw_lane_overlay(overlay, mask)
            draw_hud(overlay, lane_label, state, speed, gap, trend, width)

            cv2.putText(overlay, f"Conf: {avg_conf:.4f}", (20, 125),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 220, 0), 2)

            prob_small = cv2.resize((prob_vis * 255).clip(0, 255).astype(np.uint8), (220, 110))
            prob_color = cv2.applyColorMap(prob_small, cv2.COLORMAP_JET)
            overlay[overlay.shape[0] - 116:overlay.shape[0] - 6, 6:226] = prob_color

            bev_small = cv2.resize(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), (220, 110))
            gray      = cv2.equalizeHist(cv2.cvtColor(bev_small, cv2.COLOR_BGR2GRAY))
            bev_small = cv2.cvtColor(
                cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray),
                cv2.COLOR_GRAY2BGR
            )
            bev_small = cv2.copyMakeBorder(bev_small, 4, 4, 4, 4,
                                           cv2.BORDER_CONSTANT, value=(255, 255, 255))
            bh, bw = bev_small.shape[:2]
            overlay[6:6 + bh, overlay.shape[1] - bw - 6:overlay.shape[1] - 6] = bev_small

            try:
                cv2.imshow("CARLA Night Lane XAI - Town10HD", overlay)
            except cv2.error:
                if frame_count % 30 == 0:
                    out_path = f"xai_output/frame_{frame_count:06d}.jpg"
                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        try:
            key = cv2.waitKey(1)
        except cv2.error:
            key = -1
        if key == 27:
            break

finally:
    print("\nCleaning up CARLA actors safely...")
    shutdown_flag = True

    try:
        camera.stop()
    except Exception:
        pass

    time.sleep(0.3)

    try:
        settings.synchronous_mode    = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
    except Exception:
        pass

    try:
        traffic_manager.set_synchronous_mode(False)
    except Exception:
        pass

    # Stop walker AI controllers before destroying them
    for ctrl in npc_walker_ai:
        try:
            ctrl.stop()
        except Exception:
            pass

    actors_to_destroy = []
    for actor in list(npc_walker_ai) + list(npc_walkers) + list(npc_vehicles) + [camera, vehicle]:
        try:
            if actor is not None:
                actors_to_destroy.append(carla.command.DestroyActor(actor))
        except Exception:
            pass

    if actors_to_destroy:
        try:
            client.apply_batch(actors_to_destroy)
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print("✅ Cleanup completed. Goodbye!")
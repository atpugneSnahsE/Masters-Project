"""
Video Lane Analysis — road_clip2
Fixed BEV calibration + edge masking to remove guardrail/barrier false peaks.
Added: lane centre deviation bar.
"""

import os, sys, subprocess
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_VIDEO  = os.path.expanduser("~/Downloads/road_clip2_test.mp4")
OUTPUT_VIDEO = os.path.expanduser("~/Downloads/road_clip2_out.mp4")
MODEL_PATH   = os.path.expanduser("~/Downloads/vil100_model/best_model.pth")

PROCESS_FPS  = 10
START_SEC    = 0
END_SEC      = 9999
IMG_H, IMG_W = 368, 640
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_THRESH = 0.38

# ─────────────────────────────────────────────
# BEV CALIBRATION — fixed for road_clip2
# Horizon moved down to y=215 where road edges
# are clearly separated (~180px apart).
# Bottom points from calibration tool: x=30, x=620
# ─────────────────────────────────────────────
SRC = np.float32([
    [270, 195],   # TL - left road edge near horizon
    [376, 195],   # TR - right road edge near horizon
    [634, 341],   # BR - right road edge bottom
    [ 7, 370],   # BL - left road edge bottom
])
BEV_W, BEV_H = 640, 480
DST = np.float32([
    [0,     0],
    [BEV_W, 0],
    [BEV_W, BEV_H],
    [0,     BEV_H],
])
M     = cv2.getPerspectiveTransform(SRC, DST)
M_INV = cv2.getPerspectiveTransform(DST, SRC)

# Real-world scale: 3-lane Russian highway ~11m wide
YM_PER_PIX = 30.0 / BEV_H
XM_PER_PIX = 11.0 / BEV_W

# ─────────────────────────────────────────────
# SLIDING WINDOW PARAMS
# ─────────────────────────────────────────────
N_WINDOWS  = 12
WIN_WIDTH  = 55
MIN_PIX    = 35
SMOOTH_N   = 10

# Visual
FILL_ALPHA   = 0.30
EGO_COLOR    = (0, 200, 70)
OTHER_COLORS = [(200, 40, 40), (160, 25, 25)]
BORDER_COLOR = (255, 255, 255)
BORDER_THICK = 2

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
print("Loading model...")
model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()
print(f"Model on {DEVICE}")

tf = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ─────────────────────────────────────────────
# BINARY BEV MASK
# ─────────────────────────────────────────────
def get_binary_bev(frame_rgb):
    # Model mask
    aug = tf(image=frame_rgb)
    with torch.no_grad():
        prob = torch.sigmoid(
            model(aug["image"].unsqueeze(0).to(DEVICE)).squeeze()
        ).cpu().numpy()
    model_mask = (prob > MODEL_THRESH).astype(np.uint8) * 255
    model_bev  = cv2.warpPerspective(model_mask, M, (BEV_W, BEV_H))
    model_bin  = (model_bev > 127).astype(np.uint8)

    # BEV frame for color + gradient
    bev_bgr = cv2.warpPerspective(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), M, (BEV_W, BEV_H))
    hls  = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]
    h_ch = hls[:, :, 0]

    # White lines
    white_bin  = (l_ch > 170).astype(np.uint8)
    # Yellow lines
    yellow_bin = ((h_ch > 15) & (h_ch < 35) & (s_ch > 65)).astype(np.uint8)
    color_bin  = np.clip(white_bin + yellow_bin, 0, 1)

    # Sobel X on L channel
    sx      = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=5)
    asx     = np.abs(sx)
    sc      = np.uint8(255 * asx / (asx.max() + 1e-6))
    sob_bin = ((sc >= 20) & (sc <= 200)).astype(np.uint8)

    combined = np.clip(model_bin + color_bin + sob_bin, 0, 1).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


# ─────────────────────────────────────────────
# HISTOGRAM PEAK DETECTION
# Edge masking removes guardrail + noise barrier
# ─────────────────────────────────────────────
def find_base_peaks(binary):
    hist = binary[BEV_H // 2:, :].sum(axis=0).astype(float)

    # Blank edges — guardrail (left) and noise barrier (right)
    # produce strong false peaks near x=0 and x=BEV_W
    EDGE_MARGIN = 50
    hist[:EDGE_MARGIN]         = 0
    hist[BEV_W - EDGE_MARGIN:] = 0

    mid       = BEV_W // 2
    bases     = []
    hist_copy = hist.copy()

    # Strongest peak each side first
    for side_hist, offset in [(hist_copy[:mid], 0), (hist_copy[mid:], mid)]:
        if side_hist.max() > 5:
            pk = int(np.argmax(side_hist)) + offset
            bases.append(pk)
            lo = max(0, pk - 40); hi = min(BEV_W, pk + 40)
            hist_copy[lo:hi] = 0

    # Up to 4 more peaks for multi-lane roads
    for _ in range(4):
        if hist_copy.max() < 5:
            break
        pk = int(np.argmax(hist_copy))
        bases.append(pk)
        lo = max(0, pk - 40); hi = min(BEV_W, pk + 40)
        hist_copy[lo:hi] = 0

    return sorted(bases)


# ─────────────────────────────────────────────
# SLIDING WINDOW
# ─────────────────────────────────────────────
def sliding_window(binary, base_x):
    win_h = BEV_H // N_WINDOWS
    cur_x = base_x
    all_x, all_y = [], []

    for w in range(N_WINDOWS):
        y_lo = BEV_H - (w + 1) * win_h
        y_hi = BEV_H - w * win_h
        x_lo = max(0,     cur_x - WIN_WIDTH)
        x_hi = min(BEV_W, cur_x + WIN_WIDTH)
        nzy, nzx = np.nonzero(binary[y_lo:y_hi, x_lo:x_hi])
        nzy += y_lo; nzx += x_lo
        all_x.extend(nzx.tolist())
        all_y.extend(nzy.tolist())
        if len(nzx) >= MIN_PIX:
            cur_x = int(np.mean(nzx))

    if len(all_x) < 20:
        return None

    yn = np.array(all_y, dtype=float) / BEV_H
    xn = np.array(all_x, dtype=float)
    try:
        a, b, c = np.polyfit(yn, xn, 2)
    except Exception:
        return None
    if abs(a) > 500:
        try:
            b, c = np.polyfit(yn, xn, 1); a = 0.0
        except:
            return None
    bot_x = a + b + c
    if not (-BEV_W * 0.3 <= bot_x <= BEV_W * 1.3):
        return None
    return dict(a=float(a), b=float(b), c=float(c),
                base_x=float(base_x), bot_x=float(bot_x))


def eval_lane(lane, row):
    yn = row / BEV_H
    return int(np.clip(lane["a"]*yn**2 + lane["b"]*yn + lane["c"], 0, BEV_W - 1))


# ─────────────────────────────────────────────
# CURVATURE & OFFSET
# ─────────────────────────────────────────────
def curvature(lane):
    scale_y = BEV_H * YM_PER_PIX
    a_m = lane["a"] * XM_PER_PIX / (scale_y ** 2)
    b_m = lane["b"] * XM_PER_PIX / scale_y
    denom = abs(2 * a_m)
    if denom < 1e-6:
        return None
    num = (1 + (2 * a_m * scale_y + b_m) ** 2) ** 1.5
    return num / denom


def offset_metres(ego_l, ego_r):
    lane_centre = (ego_l["bot_x"] + ego_r["bot_x"]) / 2
    return (BEV_W / 2 - lane_centre) * XM_PER_PIX


def lane_width_ok(ego_l, ego_r):
    w = abs(ego_r["bot_x"] - ego_l["bot_x"])
    return 50 < w < BEV_W * 0.70


# ─────────────────────────────────────────────
# SMOOTHER (EMA bucket-based)
# ─────────────────────────────────────────────
class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.n = n
        self.buckets = {}
        self.nxt = 0

    def _match(self, bot_x):
        best_id, best_d = None, 9999
        for bid, h in self.buckets.items():
            if not h:
                continue
            d = abs(np.mean([x["bot_x"] for x in h]) - bot_x)
            if d < best_d and d < 90:
                best_d, best_id = d, bid
        if best_id is None:
            best_id = self.nxt
            self.nxt += 1
            self.buckets[best_id] = deque(maxlen=self.n)
        return best_id

    def update(self, raw):
        seen = set()
        for lane in raw:
            bid = self._match(lane["bot_x"])
            self.buckets[bid].append(lane)
            seen.add(bid)
        out = []
        for bid, h in self.buckets.items():
            if not h:
                continue
            if bid not in seen and len(h) < 2:
                continue
            out.append(dict(
                a=float(np.mean([x["a"] for x in h])),
                b=float(np.mean([x["b"] for x in h])),
                c=float(np.mean([x["c"] for x in h])),
                base_x=float(np.mean([x["base_x"] for x in h])),
                bot_x=float(np.mean([x["bot_x"] for x in h])),
                active=bid in seen))
        active   = [l for l in out if l["active"]]
        inactive = [l for l in out if not l["active"]]
        out = (active + inactive)[:6]
        out.sort(key=lambda l: l["bot_x"])
        return out


# ─────────────────────────────────────────────
# DEVIATION BAR
# ─────────────────────────────────────────────
def draw_deviation_bar(img, offset_m, max_m=1.5):
    BAR_W, BAR_H = 200, 14
    BAR_X = IMG_W // 2 - BAR_W // 2
    BAR_Y = IMG_H - 28

    # Background
    cv2.rectangle(img, (BAR_X, BAR_Y),
                  (BAR_X + BAR_W, BAR_Y + BAR_H), (60, 60, 60), -1)
    # Centre tick
    cx = BAR_X + BAR_W // 2
    cv2.rectangle(img, (cx - 1, BAR_Y), (cx + 1, BAR_Y + BAR_H),
                  (255, 255, 255), -1)

    # Indicator (positive offset = right of centre)
    norm  = np.clip(offset_m / max_m, -1.0, 1.0)
    ind_x = int(cx - norm * (BAR_W // 2))
    ind_x = int(np.clip(ind_x, BAR_X + 4, BAR_X + BAR_W - 4))
    color = (0, 220, 255) if abs(offset_m) < 0.3 else (0, 120, 255)
    cv2.rectangle(img,
                  (ind_x - 5, BAR_Y + 2),
                  (ind_x + 5, BAR_Y + BAR_H - 2),
                  color, -1)

    cv2.putText(img, "L", (BAR_X - 14, BAR_Y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(img, "R", (BAR_X + BAR_W + 4, BAR_Y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────
def render(frame_rgb, lanes):
    if len(lanes) < 2:
        return frame_rgb

    rows    = np.arange(0, BEV_H, 2, dtype=int)
    car_x   = BEV_W / 2
    regions = list(zip(lanes[:-1], lanes[1:]))
    n_lanes = len(regions)

    # Find ego lane
    ego_idx = None
    for i, (l, r) in enumerate(regions):
        if l["bot_x"] <= car_x <= r["bot_x"]:
            ego_idx = i
            break
    if ego_idx is None and regions:
        dists   = [abs((l["bot_x"] + r["bot_x"]) / 2 - car_x) for l, r in regions]
        ego_idx = int(np.argmin(dists))

    ego_l = lanes[ego_idx]     if ego_idx is not None else None
    ego_r = lanes[ego_idx + 1] if ego_idx is not None else None

    # BEV fill
    bev_fill = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    for i, (left, right) in enumerate(regions):
        lpts = [(eval_lane(left,  r), int(r)) for r in rows]
        rpts = [(eval_lane(right, r), int(r)) for r in rows]
        poly = np.array(lpts + rpts[::-1], dtype=np.int32)
        if i == ego_idx:
            color = EGO_COLOR
        else:
            ci    = min(abs(i - (ego_idx or 0)) - 1, len(OTHER_COLORS) - 1)
            color = OTHER_COLORS[ci]
        cv2.fillPoly(bev_fill, [poly], color)

    for lane in lanes:
        pts = [(eval_lane(lane, r), int(r)) for r in rows[::4]]
        for j in range(len(pts) - 1):
            cv2.line(bev_fill, pts[j], pts[j + 1], BORDER_COLOR, BORDER_THICK)

    fill_cam = cv2.warpPerspective(bev_fill, M_INV, (IMG_W, IMG_H))
    nz       = fill_cam.sum(axis=2) > 0
    result   = frame_rgb.copy()
    result[nz] = cv2.addWeighted(frame_rgb, 1 - FILL_ALPHA,
                                  fill_cam, FILL_ALPHA, 0)[nz]

    # Info bar
    cv2.rectangle(result, (0, 0), (IMG_W, 58), (0, 0, 0), -1)

    off_val = None
    if ego_idx is not None and ego_l and ego_r and lane_width_ok(ego_l, ego_r):
        lane_num = ego_idx + 1
        off_val  = offset_metres(ego_l, ego_r)
        curv_l   = curvature(ego_l)
        curv_r   = curvature(ego_r)
        side     = "R" if off_val > 0 else "L"

        if curv_l and curv_r:
            curv     = (curv_l + curv_r) / 2
            curv_str = f"{min(curv, 9999):.0f} m"
        else:
            curv_str = "straight"

        l1 = f"Lane {lane_num} of {n_lanes}   |   Curve: {curv_str}"
        l2 = f"Offset: {abs(off_val)*100:.1f} cm {side} of lane centre"
    else:
        l1 = "Lane: detecting..."
        l2 = ""

    cv2.putText(result, l1, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
    if l2:
        cv2.putText(result, l2, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Deviation bar
    if off_val is not None:
        draw_deviation_bar(result, off_val)

    return result


# ─────────────────────────────────────────────
# VIDEO LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Cannot open: {INPUT_VIDEO}"); sys.exit(1)

src_fps    = cap.get(cv2.CAP_PROP_FPS)
frame_step = max(1, int(round(src_fps / PROCESS_FPS)))
sf         = int(START_SEC * src_fps)
ef         = min(int(END_SEC * src_fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(f"Source FPS: {src_fps:.1f}  →  every {frame_step} frames  →  output {PROCESS_FPS} fps")

pipe = subprocess.Popen([
    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{IMG_W}x{IMG_H}", "-r", str(PROCESS_FPS),
    "-i", "-", "-an", "-vcodec", "libx264", "-crf", "20",
    "-pix_fmt", "yuv420p", OUTPUT_VIDEO
], stdin=subprocess.PIPE)
print(f"→ {OUTPUT_VIDEO}")

smoother  = Smoother()
processed = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
frame_no = sf

while frame_no < ef:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(cv2.resize(frame, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
    binary    = get_binary_bev(frame_rgb)
    bases     = find_base_peaks(binary)
    raw       = [r for r in (sliding_window(binary, b) for b in bases) if r]
    lanes     = smoother.update(raw)
    annotated = render(frame_rgb, lanes)

    pipe.stdin.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR).tobytes())
    processed += 1
    frame_no  += frame_step
    print(f"frame {processed:4d}  bases={len(bases)}  lanes={len(lanes)}  "
          f"[{frame_no}/{ef}]", end="\r")

cap.release()
pipe.stdin.close()
pipe.wait()
print(f"\nDone! {processed} frames → {OUTPUT_VIDEO}")
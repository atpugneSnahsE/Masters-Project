"""
Video Lane Analysis — Sliding Window BEV Pipeline
Based on: Advanced Lane Finding (Mithi, Medium)
https://medium.com/@mithi/advanced-lane-finding-using-computer-vision-techniques-7f3230b6c6f2

Pipeline:
  1. Perspective warp to BEV
  2. Binary: model mask + HLS color + Sobel gradient
  3. Histogram peak detection for lane bases
  4. Sliding window search up the BEV mask
  5. Quadratic polynomial fit
  6. Curvature radius + centre offset
  7. Fill in BEV → warp back to camera
  8. Temporal smoothing
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
INPUT_VIDEO  = os.path.expanduser("~/Downloads/road_clip1.mp4")
OUTPUT_VIDEO = os.path.expanduser("~/Downloads/road_clip1_final.mp4")
MODEL_PATH   = os.path.expanduser("~/Downloads/vil100_model/best_model.pth")

PROCESS_FPS  = 5
START_SEC    = 10
END_SEC      = 300
IMG_H, IMG_W = 368, 640
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_THRESH = 0.38

# ─────────────────────────────────────────────
# PERSPECTIVE — verified on actual road markings
# ─────────────────────────────────────────────
# Calibration: verified on clean frame — BEV shows dashed centre line
# perfectly vertical, both lanes visible, road surface flat
# Measured on 640x368, scaled to any resolution
REF_W, REF_H = 640, 368
SRC = np.float32([
    [248/REF_W*IMG_W, 218/REF_H*IMG_H],   # TL - left road line near horizon
    [402/REF_W*IMG_W, 218/REF_H*IMG_H],   # TR - right road line near horizon
    [610/REF_W*IMG_W, 360/REF_H*IMG_H],   # BR - right road edge bottom
    [ 50/REF_W*IMG_W, 360/REF_H*IMG_H],   # BL - left road edge bottom
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
YM_PER_PIX = 30.0 / BEV_H
XM_PER_PIX = 3.7  / (BEV_W * 0.6)

# ─────────────────────────────────────────────
# SLIDING WINDOW
# ─────────────────────────────────────────────
N_WINDOWS = 12
WIN_WIDTH = 60
MIN_PIX   = 40
SMOOTH_N  = 8

# Visual
FILL_ALPHA   = 0.28
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

tf = A.Compose([A.Resize(IMG_H, IMG_W),
                A.Normalize(mean=(0.485,0.456,0.406),
                            std=(0.229,0.224,0.225)),
                ToTensorV2()])


# ─────────────────────────────────────────────
# STEP 1: BINARY BEV MASK
# model + HLS color + Sobel gradient
# ─────────────────────────────────────────────
def get_binary_bev(frame_rgb):
    # Model mask on camera frame → warp to BEV
    aug = tf(image=frame_rgb)
    with torch.no_grad():
        prob = torch.sigmoid(
            model(aug["image"].unsqueeze(0).to(DEVICE)).squeeze()
        ).cpu().numpy()
    model_mask = (prob > MODEL_THRESH).astype(np.uint8) * 255
    model_bev  = cv2.warpPerspective(model_mask, M, (BEV_W, BEV_H))
    model_bin  = (model_bev > 127).astype(np.uint8)

    # Warp frame to BEV for color/gradient
    bev_bgr = cv2.warpPerspective(
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), M, (BEV_W, BEV_H))
    hls   = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
    l_ch  = hls[:,:,1]
    s_ch  = hls[:,:,2]
    h_ch  = hls[:,:,0]

    # White lines
    white_bin  = (l_ch > 175).astype(np.uint8)
    # Yellow lines
    yellow_bin = ((h_ch > 15) & (h_ch < 35) & (s_ch > 70)).astype(np.uint8)
    color_bin  = np.clip(white_bin + yellow_bin, 0, 1)

    # Sobel X on L channel
    sx      = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=5)
    asx     = np.abs(sx)
    sc      = np.uint8(255 * asx / (asx.max() + 1e-6))
    sob_bin = ((sc >= 20) & (sc <= 200)).astype(np.uint8)

    combined = np.clip(model_bin + color_bin + sob_bin, 0, 1).astype(np.uint8)
    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined


# ─────────────────────────────────────────────
# STEP 2: HISTOGRAM PEAK DETECTION
# ─────────────────────────────────────────────
def find_base_peaks(binary):
    hist = binary[BEV_H//2:, :].sum(axis=0).astype(float)
    mid  = BEV_W // 2
    bases, hist_copy = [], hist.copy()
    # Left and right strongest peaks first
    for side_hist, offset in [(hist_copy[:mid], 0), (hist_copy[mid:], mid)]:
        if side_hist.max() > 5:
            bases.append(int(np.argmax(side_hist)) + offset)
            lo = max(0, bases[-1]-40); hi = min(BEV_W, bases[-1]+40)
            hist_copy[lo:hi] = 0
    # Extra peaks
    for _ in range(3):
        if hist_copy.max() < 5: break
        pk = int(np.argmax(hist_copy))
        bases.append(pk)
        lo = max(0, pk-40); hi = min(BEV_W, pk+40)
        hist_copy[lo:hi] = 0
    return sorted(bases)


# ─────────────────────────────────────────────
# STEP 3: SLIDING WINDOW SEARCH
# ─────────────────────────────────────────────
def sliding_window(binary, base_x):
    win_h = BEV_H // N_WINDOWS
    cur_x = base_x
    all_x, all_y = [], []

    for w in range(N_WINDOWS):
        y_lo = BEV_H - (w+1)*win_h
        y_hi = BEV_H - w*win_h
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
        try: b, c = np.polyfit(yn, xn, 1); a = 0.0
        except: return None
    bot_x = a + b + c
    if not (-BEV_W*0.3 <= bot_x <= BEV_W*1.3):
        return None
    return dict(a=float(a), b=float(b), c=float(c),
                base_x=float(base_x), bot_x=float(bot_x))


def eval_lane(l, row):
    yn = row / BEV_H
    return int(np.clip(l["a"]*yn**2 + l["b"]*yn + l["c"], 0, BEV_W-1))


# ─────────────────────────────────────────────
# CURVATURE & OFFSET
# ─────────────────────────────────────────────
def curvature(lane):
    # Evaluate at bottom of image (y_norm = 1.0)
    # Convert from pixel space to metres
    # x_pixels = a*y_norm^2 + b*y_norm + c
    # x_metres = x_pixels * XM_PER_PIX
    # y_metres = y_norm * BEV_H * YM_PER_PIX
    # So: a_m = a * XM_PER_PIX / (BEV_H * YM_PER_PIX)^2
    #     b_m = b * XM_PER_PIX / (BEV_H * YM_PER_PIX)
    scale_y = BEV_H * YM_PER_PIX   # total metres in BEV height
    a_m = lane["a"] * XM_PER_PIX / (scale_y ** 2)
    b_m = lane["b"] * XM_PER_PIX / scale_y
    denom = abs(2 * a_m)
    if denom < 1e-6: return None   # straight line
    num = (1 + (2 * a_m * scale_y + b_m) ** 2) ** 1.5
    return num / denom

def offset(ego_l, ego_r):
    return (BEV_W/2 - (ego_l["bot_x"] + ego_r["bot_x"])/2) * XM_PER_PIX

def lane_width_ok(ego_l, ego_r):
    w = abs(ego_r["bot_x"] - ego_l["bot_x"])
    return 60 < w < BEV_W * 0.65


# ─────────────────────────────────────────────
# SMOOTHER
# ─────────────────────────────────────────────
class Smoother:
    def __init__(self, n=SMOOTH_N):
        self.n = n; self.buckets = {}; self.nxt = 0

    def _match(self, bot_x):
        best_id, best_d = None, 9999
        for bid, h in self.buckets.items():
            if not h: continue
            d = abs(np.mean([x["bot_x"] for x in h]) - bot_x)
            if d < best_d and d < 90: best_d, best_id = d, bid
        if best_id is None:
            best_id = self.nxt; self.nxt += 1
            self.buckets[best_id] = deque(maxlen=self.n)
        return best_id

    def update(self, raw):
        seen = set()
        for lane in raw:
            bid = self._match(lane["bot_x"])
            self.buckets[bid].append(lane); seen.add(bid)
        out = []
        for bid, h in self.buckets.items():
            if not h: continue
            if bid not in seen and len(h) < 2: continue
            out.append(dict(
                a=float(np.mean([x["a"] for x in h])),
                b=float(np.mean([x["b"] for x in h])),
                c=float(np.mean([x["c"] for x in h])),
                base_x=float(np.mean([x["base_x"] for x in h])),
                bot_x=float(np.mean([x["bot_x"] for x in h])),
                active=bid in seen))
        active = [l for l in out if l["active"]]
        inactive = [l for l in out if not l["active"]]
        out = (active + inactive)[:5]
        out.sort(key=lambda l: l["bot_x"])
        return out


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────
def render(frame_rgb, lanes):
    if len(lanes) < 2:
        return frame_rgb

    rows = np.arange(0, BEV_H, 2, dtype=int)

    # Ego lane = the region that contains BEV centre x at the bottom
    # The car camera is at the centre of the BEV image
    car_x   = BEV_W / 2
    regions = list(zip(lanes[:-1], lanes[1:]))
    ego_idx = None
    for i, (l, r) in enumerate(regions):
        if l["bot_x"] <= car_x <= r["bot_x"]:
            ego_idx = i
            break
    # Fallback: closest region to centre
    if ego_idx is None and regions:
        dists   = [abs((l["bot_x"]+r["bot_x"])/2 - car_x) for l,r in regions]
        ego_idx = int(np.argmin(dists))

    ego_l = lanes[ego_idx]     if ego_idx is not None else None
    ego_r = lanes[ego_idx+1]   if ego_idx is not None else None
    left_l = [l for l in lanes if l["bot_x"] < car_x]

    bev_fill = np.zeros((BEV_H, BEV_W, 3), dtype=np.uint8)
    for i,(left,right) in enumerate(regions):
        lpts = [(eval_lane(left, r), int(r)) for r in rows]
        rpts = [(eval_lane(right,r), int(r)) for r in rows]
        poly = np.array(lpts+rpts[::-1], dtype=np.int32)
        color = EGO_COLOR if i==ego_idx else \
                OTHER_COLORS[min(abs(i-(ego_idx or 0))-1, len(OTHER_COLORS)-1)]
        cv2.fillPoly(bev_fill, [poly], color)

    for lane in lanes:
        pts = [(eval_lane(lane, r), int(r)) for r in rows[::4]]
        for j in range(len(pts)-1):
            cv2.line(bev_fill, pts[j], pts[j+1], BORDER_COLOR, BORDER_THICK)

    fill_cam = cv2.warpPerspective(bev_fill, M_INV, (IMG_W, IMG_H))
    nz       = fill_cam.sum(axis=2) > 0
    result   = frame_rgb.copy()
    result[nz] = cv2.addWeighted(frame_rgb,1-FILL_ALPHA,fill_cam,FILL_ALPHA,0)[nz]

    n = len(lanes)
    if ego_l and ego_r and lane_width_ok(ego_l, ego_r):
        off    = offset(ego_l, ego_r)
        curv_l = curvature(ego_l)
        curv_r = curvature(ego_r)
        side   = "R" if off > 0 else "L"
        if curv_l and curv_r:
            curv = (curv_l + curv_r) / 2
            curv_str = f"{min(curv, 9999):.0f}m"
        else:
            curv_str = "straight"
        l1 = f"Lane {len(left_l)} of {n-1}   Curve: {curv_str}"
        l2 = f"Offset: {abs(off)*100:.1f}cm {side} of centre"
    elif not ego_l: l1,l2 = "Leftmost lane",""
    else:           l1,l2 = "Rightmost lane",""

    cv2.rectangle(result,(0,0),(IMG_W,52),(0,0,0),-1)
    cv2.putText(result,l1,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
    if l2: cv2.putText(result,l2,(8,44),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
    return result


# ─────────────────────────────────────────────
# VIDEO LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened(): print(f"Cannot open: {INPUT_VIDEO}"); sys.exit(1)

src_fps    = cap.get(cv2.CAP_PROP_FPS)
frame_step = max(1,int(round(src_fps/PROCESS_FPS)))
sf         = int(START_SEC*src_fps)
ef         = min(int(END_SEC*src_fps),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print(f"FPS:{src_fps:.1f} step:{frame_step}")

pipe = subprocess.Popen([
    "ffmpeg","-y","-f","rawvideo","-pix_fmt","bgr24",
    "-s",f"{IMG_W}x{IMG_H}","-r",str(PROCESS_FPS),
    "-i","-","-an","-vcodec","libx264","-crf","20",
    "-pix_fmt","yuv420p",OUTPUT_VIDEO],stdin=subprocess.PIPE)
print("→",OUTPUT_VIDEO)

smoother = Smoother()
processed = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
frame_no = sf

while frame_no < ef:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret: break
    frame_rgb = cv2.cvtColor(cv2.resize(frame,(IMG_W,IMG_H)),cv2.COLOR_BGR2RGB)
    binary    = get_binary_bev(frame_rgb)
    bases     = find_base_peaks(binary)
    raw       = [r for r in (sliding_window(binary,b) for b in bases) if r]
    lanes     = smoother.update(raw)
    annotated = render(frame_rgb, lanes)
    pipe.stdin.write(cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR).tobytes())
    processed += 1; frame_no += frame_step
    print(f"frame {processed:4d} bases={len(bases)} lanes={len(lanes)}",end="\r")

cap.release(); pipe.stdin.close(); pipe.wait()
print(f"\nDone! {processed} frames → {OUTPUT_VIDEO}")
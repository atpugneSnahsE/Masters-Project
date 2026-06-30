"""
ADAS Pipeline v4
================
COMPLETELY NEW LANE LOGIC:
  - U-Net gives us 6-class mask (bg + line_1..line_5)
  - For each detected class, extract its spine (median x per row)
  - Fit a polynomial to that spine
  - Find which two adjacent class lines bracket the image center → ego lane
  - Fill the polygon BETWEEN those two fitted spines directly
  - No scanning, no peak detection, no clustering — we trust the mask

This means the green fill will always follow the actual detected markings.
"""

import os, sys, subprocess
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp
import albumentations as A
import warnings
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = os.path.expanduser("~/Downloads/lithuania_road.mp4")
OUTPUT_VIDEO = os.path.expanduser("~/Downloads/lithuania_road_out.mp4")
LANE_MODEL   = os.path.expanduser("~/Downloads/best_lane_model.pth")
YOLO_MODEL   = os.path.expanduser("~/Downloads/traffic_best.pt")

PROCESS_FPS  = 10
START_SEC    = 0
END_SEC      = 9999
IMG_H, IMG_W = 368, 640
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES    = 6        # background(0) + line_1(1)..line_5(5)
MODEL_THRESH = 0.45     # min softmax prob for a pixel to count as a lane class

ROI_TOP_FRAC = 0.45     # ignore sky above this fraction
MIN_ROWS_FOR_FIT = 8    # minimum rows where a class is visible to attempt fit

EMA_ALPHA    = 0.35     # lane smoothing

FILL_ALPHA   = 0.40
EGO_COLOR    = (0, 200, 70)   # green fill (RGB)

# Per-class overlay colors (RGB)
CLASS_RGB = {
    1: (255,  60,  60),   # line_1 — red
    2: (255, 220,   0),   # line_2 — yellow
    3: ( 60, 220,  60),   # line_3 — green
    4: (  0, 220, 220),   # line_4 — cyan
    5: ( 60,  60, 255),   # line_5 — blue
}
SHOW_CLASS_OVERLAY = True

# BEV (for curvature / offset)
BEV_W, BEV_H = 640, 480
YM_PER_PIX   = 30.0 / BEV_H
XM_PER_PIX   =  3.7 / BEV_W

# YOLO
YOLO_EVERY    = 8
YOLO_CONF     = 0.30
MARKING_IDS   = set(range(29, 41))
SIGN_COLOR    = (255, 200,   0)
MARKING_COLOR = (  0, 200, 255)

ROI_TOP = int(ROI_TOP_FRAC * IMG_H)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("Loading lane model (ResNet34 U-Net, 6-class)…")
lane_model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=N_CLASSES,
)
lane_model.load_state_dict(torch.load(LANE_MODEL, map_location=DEVICE))
lane_model.to(DEVICE).eval()
print(f"  Lane model on {DEVICE}")

print("Loading YOLO…")
yolo = YOLO(YOLO_MODEL)
print(f"  YOLO: {len(yolo.names)} classes")

tf = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RUN U-NET → class mask
# ─────────────────────────────────────────────────────────────────────────────
def get_class_mask(frame_rgb):
    """
    Returns:
      class_mask  — H×W uint8, values 0-5 (argmax of softmax)
      conf_mask   — H×W float32, confidence of the winning class
    """
    aug = tf(image=frame_rgb)
    with torch.no_grad():
        logits = lane_model(aug["image"].unsqueeze(0).to(DEVICE))   # (1,6,H,W)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (6,H,W)

    class_mask = probs.argmax(axis=0).astype(np.uint8)
    conf_mask  = probs.max(axis=0)

    # Zero out low-confidence pixels and sky
    class_mask[conf_mask < MODEL_THRESH] = 0
    class_mask[:ROI_TOP, :] = 0

    return class_mask, conf_mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT SPINE PER CLASS
# ─────────────────────────────────────────────────────────────────────────────
def extract_spine(class_mask, cls_id):
    """
    For a given class, find the median x-position of its pixels per row.
    Returns list of (row, x) pairs for rows where the class is visible.
    """
    rows_with_class = np.where(
        (class_mask == cls_id).any(axis=1)
    )[0]
    rows_with_class = rows_with_class[rows_with_class >= ROI_TOP]

    spine = []
    for row in rows_with_class:
        xs = np.where(class_mask[row] == cls_id)[0]
        if len(xs) > 0:
            spine.append((row, int(np.median(xs))))
    return spine


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FIT POLYNOMIAL TO SPINE
# ─────────────────────────────────────────────────────────────────────────────
def fit_spine(spine):
    """
    Fit x = f(y) quadratic to the spine points.
    Returns dict with poly coeffs + bot_x/top_x, or None if too few points.
    """
    if len(spine) < MIN_ROWS_FOR_FIT:
        return None

    ys = np.array([p[0] for p in spine], dtype=float)
    xs = np.array([p[1] for p in spine], dtype=float)
    yn = ys / IMG_H   # normalise for numerical stability

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            coeffs = np.polyfit(yn, xs, 2)
        except Exception:
            try:
                b, c   = np.polyfit(yn, xs, 1)
                coeffs = (0.0, b, c)
            except Exception:
                return None

    a, b, c = coeffs
    bot_x = a + b + c            # yn=1.0 → y=IMG_H
    top_x = (a*(ROI_TOP/IMG_H)**2
              + b*(ROI_TOP/IMG_H) + c)

    # Sanity: must land somewhere plausible horizontally
    if not (-IMG_W * 0.4 <= bot_x <= IMG_W * 1.4):
        return None

    # Reject wildly curving fits — top/bottom spread > 55% of image width
    # means the quadratic has overshot and will cause X-crossings.
    # Fall back to linear which cannot overshoot.
    if abs(top_x - bot_x) > IMG_W * 0.55:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                b2, c2 = np.polyfit(yn, xs, 1)
                a, b, c = 0.0, float(b2), float(c2)
                bot_x   = b + c
                top_x   = b*(ROI_TOP/IMG_H) + c
            except Exception:
                return None

    return dict(cls_id=None, a=float(a), b=float(b), c=float(c),
                bot_x=float(bot_x), top_x=float(top_x))


def eval_line(lane, y_px):
    yn = y_px / IMG_H
    return int(np.clip(lane["a"]*yn**2 + lane["b"]*yn + lane["c"], 0, IMG_W-1))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD FITTED LINES DICT  {cls_id: lane_dict}
# ─────────────────────────────────────────────────────────────────────────────
def get_fitted_lines(class_mask):
    """
    For each visible class (1-5), extract spine and fit polynomial.
    Returns dict {cls_id: lane_dict}, sorted by bot_x (left to right).
    """
    fitted = {}
    for cls_id in range(1, N_CLASSES):
        spine = extract_spine(class_mask, cls_id)
        lane  = fit_spine(spine)
        if lane is not None:
            lane["cls_id"] = cls_id
            fitted[cls_id] = lane
    return fitted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SELECT EGO LANE PAIR
# ─────────────────────────────────────────────────────────────────────────────
def select_ego_pair(fitted):
    """
    Sort detected lines left->right by bot_x.
    Find the adjacent pair that brackets the bottom-center of the image.

    Key rules:
    1. Prefer pairs that are ADJACENT in class ID (e.g. 2&3, not 2&5)
       — skipping a class means a line is missing, fill would be too wide
    2. Max allowed fill width = 45% of image width (reject road-spanning pairs)
    3. Both lines must not cross each other at any row in the ROI
    """
    if len(fitted) < 2:
        return None, None

    cx  = IMG_W / 2.0
    MAX_FILL_WIDTH = IMG_W * 0.45

    # Sort by x position at bottom
    sorted_cls = sorted(fitted.keys(), key=lambda c: fitted[c]["bot_x"])

    # Build candidate pairs: prefer adjacent class IDs
    def pair_score(cl, cr):
        """Lower = better. Adjacent class IDs score 0, gap of 1 scores 1, etc."""
        return abs(cr - cl) - 1   # 0 for adjacent, 1 for gap-of-1, etc.

    best_pair  = None
    best_score = 999

    for i in range(len(sorted_cls) - 1):
        cl = sorted_cls[i]
        cr = sorted_cls[i + 1]
        lx = fitted[cl]["bot_x"]
        rx = fitted[cr]["bot_x"]

        # Must bracket center
        if not (lx <= cx <= rx):
            continue

        # Must not be too wide
        if (rx - lx) > MAX_FILL_WIDTH:
            continue

        # Must not cross in the ROI
        check_rows = np.linspace(ROI_TOP, IMG_H - 1, 15, dtype=int)
        crossings  = sum(1 for r in check_rows
                         if eval_line(fitted[cl], r) >= eval_line(fitted[cr], r))
        if crossings > 2:
            continue

        score = pair_score(cl, cr)
        if score < best_score:
            best_score = score
            best_pair  = (cl, cr)

    if best_pair is not None:
        return best_pair

    # Fallback: just pick the two closest lines to center that don't cross
    # and aren't too wide — don't fall back to outermost pair
    candidates = []
    for i in range(len(sorted_cls) - 1):
        cl = sorted_cls[i]
        cr = sorted_cls[i + 1]
        lx = fitted[cl]["bot_x"]
        rx = fitted[cr]["bot_x"]
        if (rx - lx) > MAX_FILL_WIDTH:
            continue
        check_rows = np.linspace(ROI_TOP, IMG_H - 1, 15, dtype=int)
        crossings  = sum(1 for r in check_rows
                         if eval_line(fitted[cl], r) >= eval_line(fitted[cr], r))
        if crossings > 2:
            continue
        dist_to_center = abs((lx + rx) / 2.0 - cx)
        candidates.append((dist_to_center, cl, cr))

    if candidates:
        candidates.sort()
        return candidates[0][1], candidates[0][2]

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EMA SMOOTHER PER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class ClassLineSmoother:
    """Smooths fitted lane coefficients per class independently."""
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha  = alpha
        self.states = {}   # {cls_id: {a,b,c,bot_x,top_x}}

    def update(self, fitted):
        """fitted = {cls_id: lane_dict}. Returns smoothed fitted dict."""
        smoothed = {}
        for cls_id, lane in fitted.items():
            if cls_id not in self.states:
                self.states[cls_id] = {k: lane[k]
                                       for k in ("a","b","c","bot_x","top_x")}
            else:
                for k in ("a","b","c","bot_x","top_x"):
                    self.states[cls_id][k] = (
                        self.alpha * lane[k]
                        + (1 - self.alpha) * self.states[cls_id][k]
                    )
            s = dict(self.states[cls_id])
            s["cls_id"] = cls_id
            smoothed[cls_id] = s

        # Decay classes not seen this frame (don't hard-reset immediately)
        for cls_id in list(self.states.keys()):
            if cls_id not in fitted:
                del self.states[cls_id]   # drop after 1 missed frame

        return smoothed

def calculate_vanishing_point(smoothed_lanes):
    """
    Finds the common intersection point (VP) of all detected lanes.
    Returns (vx, vy) or None.
    """
    if len(smoothed_lanes) < 2:
        return None

    intersections = []
    ids = list(smoothed_lanes.keys())
    
    # We use the linear part (b, c) for the horizon intersection 
    # as the quadratic 'a' often flattens out near the horizon.
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            l1, l2 = smoothed_lanes[ids[i]], smoothed_lanes[ids[j]]
            
            # Line eq: x = b*y_norm + c
            # Intersection: b1*y + c1 = b2*y + c2  => y = (c2 - c1) / (b1 - b2)
            db = l1['b'] - l2['b']
            if abs(db) < 0.001: continue # Parallel lines
            
            y_int = (l2['c'] - l1['c']) / db
            x_int = l1['b'] * y_int + l1['c']
            
            # Convert y_norm back to pixel space
            intersections.append((x_int, y_int * IMG_H))

    if not intersections:
        return None

    # Use Median to stay robust against one 'bad' line fit
    vx = np.median([p[0] for p in intersections])
    vy = np.median([p[1] for p in intersections])
    
    return int(vx), int(vy)
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — CURVATURE & OFFSET  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
def compute_dynamic_bev(ego_l, ego_r):
    if ego_l is None or ego_r is None:
        return None, None
    y_bot = float(IMG_H - 1)
    y_top = float(ROI_TOP + (IMG_H - ROI_TOP) * 0.15)
    bl = (float(eval_line(ego_l, int(y_bot))), y_bot)
    br = (float(eval_line(ego_r, int(y_bot))), y_bot)
    tl = (float(eval_line(ego_l, int(y_top))), y_top)
    tr = (float(eval_line(ego_r, int(y_top))), y_top)
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0,0],[BEV_W,0],[BEV_W,BEV_H],[0,BEV_H]])
    try:
        M     = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)
        return M, M_inv
    except Exception:
        return None, None


def curvature_and_offset(ego_l, ego_r, M):
    if M is None:
        return "straight", 0.0
    rows = np.linspace(0, IMG_H-1, 40, dtype=int)

    def warp(lane):
        pts = np.array([[eval_line(lane, r), r] for r in rows],
                       dtype=np.float32).reshape(-1,1,2)
        w = cv2.perspectiveTransform(pts, M).reshape(-1,2)
        return w[:,0], w[:,1]

    try:
        lx, ly = warp(ego_l)
        rx, ry = warp(ego_r)
    except Exception:
        return "straight", 0.0

    yn = ly / BEV_H

    def safe_fit(yn, xs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return np.polyfit(yn, xs, 2)
            except Exception:
                try:
                    b,c = np.polyfit(yn, xs, 1)
                    return np.array([0.0, b, c])
                except Exception:
                    return np.array([0.0, 0.0, xs.mean()])

    al,bl_,cl = safe_fit(yn, lx)
    ar,br_,cr = safe_fit(yn, rx)

    def curv(a, b):
        if abs(a) < 1e-6: return None
        sy = BEV_H * YM_PER_PIX
        am = a * XM_PER_PIX / sy**2
        bm = b * XM_PER_PIX / sy
        d  = abs(2*am)
        return ((1+(2*am*sy+bm)**2)**1.5)/d if d>1e-6 else None

    cl_ = curv(al,bl_); cr_ = curv(ar,br_)
    curv_str = (f"{min((cl_+cr_)/2, 9999):.0f} m"
                if cl_ and cr_ else "straight")
    lane_mid = ((al+bl_+cl)+(ar+br_+cr)) / 2.0
    offset_m = (BEV_W/2.0 - lane_mid) * XM_PER_PIX
    return curv_str, float(offset_m)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo(frame_bgr):
    results = yolo(frame_bgr, conf=YOLO_CONF, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls_id     = int(box.cls)
        conf       = float(box.conf)
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        dets.append(dict(cls_id=cls_id, conf=conf,
                         x1=x1, y1=y1, x2=x2, y2=y2,
                         name=yolo.names[cls_id],
                         is_marking=(cls_id in MARKING_IDS)))
    return dets


def draw_detections(frame_rgb, dets):
    img = frame_rgb.copy()
    for d in dets:
        color = MARKING_COLOR if d["is_marking"] else SIGN_COLOR
        cv2.rectangle(img,(d["x1"],d["y1"]),(d["x2"],d["y2"]),color,2)
        label = f"{d['name']} {d['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img,(d["x1"],d["y1"]-th-6),(d["x1"]+tw+4,d["y1"]),color,-1)
        cv2.putText(img,label,(d["x1"]+2,d["y1"]-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# DEVIATION BAR
# ─────────────────────────────────────────────────────────────────────────────
def draw_deviation_bar(img, offset_m, max_m=1.5):
    BAR_W, BAR_H = 200, 14
    BAR_X = IMG_W//2 - BAR_W//2
    BAR_Y = IMG_H - 28
    cv2.rectangle(img,(BAR_X,BAR_Y),(BAR_X+BAR_W,BAR_Y+BAR_H),(60,60,60),-1)
    cx = BAR_X + BAR_W//2
    cv2.rectangle(img,(cx-1,BAR_Y),(cx+1,BAR_Y+BAR_H),(255,255,255),-1)
    norm  = np.clip(offset_m/max_m, -1.0, 1.0)
    ind_x = int(np.clip(cx - norm*(BAR_W//2), BAR_X+4, BAR_X+BAR_W-4))
    color = (0,220,255) if abs(offset_m)<0.3 else (0,120,255)
    cv2.rectangle(img,(ind_x-5,BAR_Y+2),(ind_x+5,BAR_Y+BAR_H-2),color,-1)
    cv2.putText(img,"L",(BAR_X-14,BAR_Y+11),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    cv2.putText(img,"R",(BAR_X+BAR_W+4,BAR_Y+11),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render(frame_rgb, class_mask, smoothed_fitted,
           ego_l_cls, ego_r_cls,
           dets, curv_str, offset_m):

    result = draw_detections(frame_rgb, dets)

    # ── per-class color overlay ───────────────────────────────────────────────
    if SHOW_CLASS_OVERLAY:
        overlay = np.zeros_like(result)
        for cls_id, rgb in CLASS_RGB.items():
            overlay[class_mask == cls_id] = rgb
        result = cv2.addWeighted(result, 0.72, overlay, 0.28, 0)

    rows = np.arange(int(IMG_H * 0.60), IMG_H, 2, dtype=int)

    # ── ego lane fill — directly between the two ego class spines ────────────
    ego_l = smoothed_fitted.get(ego_l_cls)
    ego_r = smoothed_fitted.get(ego_r_cls)

    if ego_l is not None and ego_r is not None:
        # Extra safety: ensure they don't cross at the bottom
        if ego_l["bot_x"] < ego_r["bot_x"]:
            lpts = np.array([[eval_line(ego_l, r), r] for r in rows], np.int32)
            rpts = np.array([[eval_line(ego_r, r), r] for r in rows], np.int32)
            poly = np.concatenate([lpts, rpts[::-1]], axis=0)
            fill = np.zeros_like(result)
            cv2.fillPoly(fill, [poly], EGO_COLOR)
            result = cv2.addWeighted(result, 1.0, fill, FILL_ALPHA, 0)

    # ── draw all fitted class spines ──────────────────────────────────────────
    for cls_id, lane in smoothed_fitted.items():
        is_ego = cls_id in (ego_l_cls, ego_r_cls)
        color  = CLASS_RGB.get(cls_id, (255,255,255))
        thick  = 3 if is_ego else 2
        pts    = np.array([[eval_line(lane, r), r]
                           for r in rows[::3]], np.int32)
        for j in range(len(pts)-1):
            cv2.line(result, tuple(pts[j]), tuple(pts[j+1]), color, thick)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.rectangle(result, (0,0), (IMG_W,60), (0,0,0), -1)

    n_lines  = len(smoothed_fitted)
    n_left   = sum(1 for c,l in smoothed_fitted.items()
                   if l["bot_x"] < IMG_W/2)
    n_lanes  = max(n_lines - 1, 1)

    n_signs    = sum(1 for d in dets if not d["is_marking"])
    n_markings = sum(1 for d in dets if d["is_marking"])

    if ego_l is not None and ego_r is not None and ego_l["bot_x"] < ego_r["bot_x"]:
        side = "R" if offset_m > 0 else "L"
        l1   = (f"Lane {n_left} of {n_lanes}"
                f"   |   Curve: {curv_str}"
                f"   |   Offset: {abs(offset_m)*100:.1f} cm {side}")
        l2   = (f"Ego: line_{ego_l_cls} to line_{ego_r_cls}"
                f"   |   Signs:{n_signs}  Markings:{n_markings}")
    else:
        l1 = "Lane: detecting…"
        l2 = f"Signs:{n_signs}  Markings:{n_markings}"

    cv2.putText(result, l1, (10,24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(result, l2, (10,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    if (ego_l is not None and ego_r is not None
            and ego_l["bot_x"] < ego_r["bot_x"]):
        draw_deviation_bar(result, offset_m)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    print(f"Cannot open: {INPUT_VIDEO}"); sys.exit(1)

src_fps    = cap.get(cv2.CAP_PROP_FPS)
frame_step = max(1, int(round(src_fps / PROCESS_FPS)))
sf         = int(START_SEC * src_fps)
ef         = min(int(END_SEC * src_fps),
                 int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

print(f"Source FPS:{src_fps:.1f}  step:{frame_step}  out:{PROCESS_FPS}fps")

pipe = subprocess.Popen([
    "ffmpeg","-y","-f","rawvideo","-pix_fmt","bgr24",
    "-s",f"{IMG_W}x{IMG_H}","-r",str(PROCESS_FPS),
    "-i","-","-an","-vcodec","libx264","-crf","20",
    "-pix_fmt","yuv420p", OUTPUT_VIDEO
], stdin=subprocess.PIPE)
print(f"→ {OUTPUT_VIDEO}")

smoother        = ClassLineSmoother(alpha=EMA_ALPHA)
last_dets       = []
last_ego_l_cls  = None
last_ego_r_cls  = None
processed       = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
frame_no = sf

while frame_no < ef:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret: break

    frame_resized = cv2.resize(frame, (IMG_W, IMG_H))
    frame_rgb     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # 1. U-Net → class mask
    class_mask, _ = get_class_mask(frame_rgb)

    # 2. Fit spine per class
    fitted = get_fitted_lines(class_mask)

    # 3. Smooth
    smoothed = smoother.update(fitted)

    # 4. Select ego pair from smoothed lines
    ego_l_cls, ego_r_cls = select_ego_pair(smoothed)
    if ego_l_cls is not None:
        last_ego_l_cls = ego_l_cls
        last_ego_r_cls = ego_r_cls
    else:
        # Hold last known ego pair if we temporarily lose detection
        ego_l_cls = last_ego_l_cls
        ego_r_cls = last_ego_r_cls

    # 5. BEV → curvature + offset
    ego_l = smoothed.get(ego_l_cls)
    ego_r = smoothed.get(ego_r_cls)
    M, M_inv = compute_dynamic_bev(ego_l, ego_r)
    curv_str, offset_m = curvature_and_offset(ego_l, ego_r, M)

    # 6. YOLO every N frames
    if processed % YOLO_EVERY == 0:
        last_dets = run_yolo(frame_resized)

    # 7. Render
    annotated = render(
        frame_rgb, class_mask, smoothed,
        ego_l_cls, ego_r_cls,
        last_dets, curv_str, offset_m
    )

    pipe.stdin.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR).tobytes())
    processed += 1
    frame_no  += frame_step

    cls_visible = sorted(smoothed.keys())
    print(f"frame {processed:4d}  "
          f"classes={cls_visible}  "
          f"ego={ego_l_cls}→{ego_r_cls}  "
          f"curve={curv_str}  offset={offset_m:+.2f}m  "
          f"det={len(last_dets)}",
          end="\r")

cap.release()
pipe.stdin.close()
pipe.wait()
print(f"\nDone — {processed} frames → {OUTPUT_VIDEO}")
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
# CONFIG — STRICT MODE
# ─────────────────────────────────────────────────────────────────────────────
INPUT_VIDEO  = os.path.expanduser("~/Downloads/road_clip_excerpt.mp4")
OUTPUT_VIDEO = os.path.expanduser("~/Downloads/adas_out_v12.mp4")
LANE_MODEL   = os.path.expanduser("~/Downloads/best_lane_model.pth")
YOLO_MODEL   = os.path.expanduser("~/Downloads/traffic_best.pt")

PROCESS_FPS  = 10
IMG_H, IMG_W = 368, 640
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

N_CLASSES     = 6
MODEL_THRESH  = 0.60  # Raised to 0.60 to ignore noisy "haze" detections
EMA_ALPHA     = 0.40  # Faster reaction, less "lag"
MAX_LANE_WIDTH = IMG_W * 0.38 # Reject anything wider than this (Intersections)
MIN_LANE_WIDTH = 60           # Reject anything too thin (Noise)

ROI_TOP = int(0.45 * IMG_H) 
FILL_ALPHA = 0.35
EGO_COLOR  = (0, 200, 70) 

# YOLO Settings
YOLO_EVERY = 8
MARKING_IDS = set(range(29, 41))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────
lane_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=N_CLASSES)
lane_model.load_state_dict(torch.load(LANE_MODEL, map_location=DEVICE))
lane_model.to(DEVICE).eval()
yolo = YOLO(YOLO_MODEL)

tf = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def get_fitted_lines(frame_rgb):
    aug = tf(image=frame_rgb)
    with torch.no_grad():
        logits = lane_model(aug["image"].unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    class_mask = probs.argmax(axis=0).astype(np.uint8)
    conf_mask  = probs.max(axis=0)
    class_mask[conf_mask < MODEL_THRESH] = 0
    class_mask[:ROI_TOP, :] = 0

    fitted = {}
    for cid in range(1, N_CLASSES):
        rows = np.where((class_mask == cid).any(axis=1))[0]
        if len(rows) < 10: continue
        
        spine = []
        for r in rows:
            xs = np.where(class_mask[r] == cid)[0]
            spine.append((r, int(np.median(xs))))
        
        ys = np.array([p[0] for p in spine], dtype=float) / IMG_H
        xs = np.array([p[1] for p in spine], dtype=float)
        
        try:
            # Force Linear fit if road is simple to prevent "whip"
            coeffs = np.polyfit(ys, xs, 1)
            a, b, c = 0.0, float(coeffs[0]), float(coeffs[1])
            bot_x = b + c
            fitted[cid] = {"a":a, "b":b, "c":c, "bot_x": bot_x}
        except: continue
            
    return fitted, class_mask

def select_ego(fitted):
    if len(fitted) < 2: return None, None
    cx = IMG_W / 2.0
    s_ids = sorted(fitted.keys(), key=lambda i: fitted[i]["bot_x"])
    
    best_pair = (None, None)
    min_dist = 999
    
    for i in range(len(s_ids)-1):
        l, r = s_ids[i], s_ids[i+1]
        lx, rx = fitted[l]["bot_x"], fitted[r]["bot_x"]
        
        if lx < cx < rx:
            w = rx - lx
            if MIN_LANE_WIDTH < w < MAX_LANE_WIDTH:
                dist = abs((lx+rx)/2.0 - cx)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (l, r)
    return best_pair

def eval_line(l, y):
    yn = y / IMG_H
    return int(l["a"]*yn**2 + l["b"]*yn + l["c"])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(INPUT_VIDEO)
pipe = subprocess.Popen([
    "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{IMG_W}x{IMG_H}", "-r", "10",
    "-i", "-", "-an", "-vcodec", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", OUTPUT_VIDEO
], stdin=subprocess.PIPE)

last_dets = []
processed = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_res = cv2.resize(frame, (IMG_W, IMG_H))
    frame_rgb = cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB)

    # 1. Detect & Fit
    fitted, mask = get_fitted_lines(frame_rgb)
    ego_l, ego_r = select_ego(fitted)

    # 2. YOLO
    if processed % YOLO_EVERY == 0:
        results = yolo(frame_res, conf=0.3, verbose=False)[0]
        last_dets = []
        for b in results.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            last_dets.append((x1,y1,x2,y2, yolo.names[int(b.cls)]))

    # 3. Render
    out = frame_res.copy()
    rows = np.arange(ROI_TOP, IMG_H, 4, dtype=int)
    
    # Fill Lane
    if ego_l and ego_r:
        lp = np.array([[eval_line(fitted[ego_l], r), r] for r in rows], np.int32)
        rp = np.array([[eval_line(fitted[ego_r], r), r] for r in rows], np.int32)
        cv2.fillPoly(out, [np.concatenate([lp, rp[::-1]])], EGO_COLOR)
        out = cv2.addWeighted(frame_res, 0.65, out, 0.35, 0)

    # Draw Objects
    for (x1,y1,x2,y2, name) in last_dets:
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(out, name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    pipe.stdin.write(out.tobytes())
    processed += 1
    print(f"Frame {processed}", end="\r")

cap.release()
pipe.stdin.close()
pipe.wait()
print(f"\nSaved to {OUTPUT_VIDEO}")
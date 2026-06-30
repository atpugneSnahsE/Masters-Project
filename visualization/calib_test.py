"""
BEV Calibration Tool
====================
Click and drag the 4 green corners to adjust the trapezoid.
The BEV result updates live in the right panel.
When lanes look vertical and parallel → copy the printed SRC values.

Controls:
  drag green dots  — move trapezoid corners
  's'              — save / print current SRC values
  'n'              — next frame (+30 frames)
  'q'              — quit
"""

import cv2
import numpy as np

VIDEO   = "/Users/mac/Downloads/road_clip_excerpt.mp4"
IMG_W, IMG_H = 640, 368
BEV_W, BEV_H = 640, 480

# Starting SRC — adjust as needed
src = np.float32([
    [220, 225],   # TL
    [450, 225],   # TR
    [530, 345],   # BR
    [ 10, 345],   # BL
])

dst = np.float32([
    [0,     0    ],
    [BEV_W, 0    ],
    [BEV_W, BEV_H],
    [0,     BEV_H],
])

COLORS   = [(0,255,0),(0,255,0),(0,255,0),(0,255,0)]
NAMES    = ["TL","TR","BR","BL"]
RADIUS   = 10
dragging = -1   # index of point being dragged

frame_orig = None
frame_no   = 0

def get_frame(cap, n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ret, f = cap.read()
    if not ret:
        return None
    return cv2.resize(f, (IMG_W, IMG_H))

def draw_overlay(frame, src_pts):
    out = frame.copy()
    # Draw trapezoid
    pts = src_pts.astype(np.int32)
    for i in range(4):
        cv2.line(out, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,0), 2)
    # Draw corner handles
    for i, pt in enumerate(pts):
        cv2.circle(out, tuple(pt), RADIUS, (0,255,0), -1)
        cv2.putText(out, NAMES[i], (pt[0]+12, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
    return out

def compute_bev(frame, src_pts):
    M   = cv2.getPerspectiveTransform(src_pts, dst)
    bev = cv2.warpPerspective(frame, M, (BEV_W, BEV_H))
    # Draw vertical guide lines at 1/4, 1/2, 3/4 width
    for x in [BEV_W//4, BEV_W//2, 3*BEV_W//4]:
        cv2.line(bev, (x,0), (x,BEV_H), (0,200,255), 1)
    return bev

def mouse_cb(event, x, y, flags, param):
    global dragging, src
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(src):
            if abs(pt[0]-x) < RADIUS+5 and abs(pt[1]-y) < RADIUS+5:
                dragging = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging >= 0:
        src[dragging] = [float(x), float(y)]
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = -1

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"Cannot open {VIDEO}"); exit(1)

WIN = "BEV Calibration  [drag corners | s=save | n=next frame | q=quit]"
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, mouse_cb)

frame_orig = get_frame(cap, frame_no)

while True:
    if frame_orig is None:
        break

    left  = draw_overlay(frame_orig, src)
    right = compute_bev(frame_orig, src)

    # Resize BEV to same height as original for side-by-side
    right_resized = cv2.resize(right, (IMG_W, IMG_H))
    combined = np.hstack([left, right_resized])

    # Instructions
    cv2.putText(combined, "Drag corners until BEV lanes are vertical",
                (10, IMG_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow(WIN, combined)
    key = cv2.waitKey(16) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        print("\n=== Copy these SRC points into combined_v4.py ===")
        print("BEV_SRC = np.float32([")
        for i, (name, pt) in enumerate(zip(NAMES, src)):
            comma = "," if i < 3 else ""
            print(f"    [{pt[0]:.0f}, {pt[1]:.0f}]{comma}   # {name}")
        print("])")
        print("=" * 48)
    elif key == ord('n'):
        frame_no += 30
        f = get_frame(cap, frame_no)
        if f is not None:
            frame_orig = f
        else:
            frame_no -= 30

cap.release()
cv2.destroyAllWindows()
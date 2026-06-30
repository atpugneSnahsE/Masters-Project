import cv2, numpy as np
IMG_H, IMG_W = 368, 640
BEV_W, BEV_H = 640, 480
SRC = np.float32([[270,200],[420,200],[510,340],[30,340]])
DST = np.float32([[0,0],[BEV_W,0],[BEV_W,BEV_H],[0,BEV_H]])
M   = cv2.getPerspectiveTransform(SRC, DST)
cap = cv2.VideoCapture("/Users/mac/Downloads/road_clip_excerpt.mp4")
ret, frame = cap.read()
frame = cv2.resize(frame, (IMG_W, IMG_H))
# draw trapezoid on original
for i in range(4): cv2.line(frame, tuple(SRC[i].astype(int)), tuple(SRC[(i+1)%4].astype(int)), (0,255,0), 2)
bev = cv2.warpPerspective(frame, M, (BEV_W, BEV_H))
cv2.imwrite("/tmp/bev_check.png", bev)
cv2.imwrite("/tmp/orig_check.png", frame)
print("Saved to /tmp/bev_check.png and /tmp/orig_check.png")
cap.release()

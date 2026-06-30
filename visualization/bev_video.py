import cv2, numpy as np, os

IMG_H, IMG_W = 368, 640
BEV_W, BEV_H = 640, 480

SRC = np.float32([
    [240, 215],
    [420, 215],
    [620, 360],
    [ 30, 360],
])
DST = np.float32([[0,0],[BEV_W,0],[BEV_W,BEV_H],[0,BEV_H]])
M = cv2.getPerspectiveTransform(SRC, DST)

img = cv2.imread(os.path.expanduser('~/Downloads/clean_frame2.png'))
img = cv2.resize(img, (IMG_W, IMG_H))

# Draw SRC points on original
for p in SRC:
    cv2.circle(img, (int(p[0]), int(p[1])), 6, (0,255,0), -1)
cv2.imwrite(os.path.expanduser('~/Downloads/src_points.png'), img)

# Warp to BEV
bev = cv2.warpPerspective(img, M, (BEV_W, BEV_H))
# Draw centre line
cv2.line(bev, (BEV_W//2, 0), (BEV_W//2, BEV_H), (0,255,255), 2)
cv2.imwrite(os.path.expanduser('~/Downloads/bev_check.png'), bev)
print('saved src_points.png and bev_check.png')
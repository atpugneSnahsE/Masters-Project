import cv2, numpy as np, os

IMG_H, IMG_W = 368, 640
img = cv2.imread(os.path.expanduser('~/Downloads/orig_frame.png'))
img = cv2.resize(img, (IMG_W, IMG_H))

# Draw a guide line at y=340 so you know where "bottom" is
cv2.line(img, (0, 340), (IMG_W, 340), (0, 0, 255), 1)
cv2.putText(img, 'click BL and BR on this line', (10, 335),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

points = []
labels = ['TL', 'TR', 'BR', 'BL']

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        cv2.circle(img, (x,y), 5, (0,255,0), -1)
        cv2.putText(img, labels[len(points)-1], (x+6,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('img', img)
        print(f"{labels[len(points)-1]}: ({x}, {y})")
        if len(points) == 4:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, [pts.reshape(-1,1,2)], True, (0,255,0), 2)
            cv2.imshow('img', img)
            print("SRC =", points)

cv2.imshow('img', img)
cv2.setMouseCallback('img', click)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2

img = cv2.imread("data/mask/000100.png", 0)

if img is None:
    print("Image not found")
else:
    cv2.imshow("mask", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

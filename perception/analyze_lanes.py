import cv2
import numpy as np
import glob

# -------- PARAMETERS --------
PIXEL_TO_METER = 0.05   # approximate, refine later

# -------- FUNCTIONS --------
def preprocess(mask):
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel

def get_components(skel):
    num, labels = cv2.connectedComponents(skel)
    return num, labels

def compute_lengths(labels, num):
    lengths = []
    for i in range(1, num):
        length = np.sum(labels == i)
        if length > 20:
            lengths.append(length)
    return lengths

def classify_dashed(lengths):
    if len(lengths) < 2:
        return "unknown"

    lengths = sorted(lengths)
    variation = np.std(lengths)

    if variation > 10:
        return "dashed"
    else:
        return "solid"

def compute_spacing(labels, num):
    centroids = []

    for i in range(1, num):
        ys, xs = np.where(labels == i)
        if len(xs) > 20:
            centroids.append(np.mean(xs))

    centroids = sorted(centroids)

    gaps = []
    for i in range(len(centroids)-1):
        gap = centroids[i+1] - centroids[i]
        gaps.append(gap)

    return gaps

def trend(gaps):
    if len(gaps) < 2:
        return "unknown"

    diffs = np.diff(gaps)

    mean_diff = np.mean(diffs)

    if mean_diff > 1:
        return "increasing"
    elif mean_diff < -1:
        return "decreasing"
    else:
        return "constant"

# -------- MAIN --------
mask_paths = sorted(glob.glob("data/mask/*.png"))

for path in mask_paths[:100]:

    mask = cv2.imread(path, 0)

    mask = preprocess(mask)

    skel = skeletonize(mask)

    num, labels = get_components(skel)

    lengths = compute_lengths(labels, num)

    lane_type = classify_dashed(lengths)

    gaps = compute_spacing(labels, num)

    spacing_m = [g * PIXEL_TO_METER for g in gaps]

    spacing_trend = trend(gaps)

    print(path)
    print("Type:", lane_type)
    print("Spacing (m):", spacing_m[:5])
    print("Trend:", spacing_trend)
    print("----")

    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis[skel > 0] = [0,255,0]

    cv2.imshow("Lane", vis)
    if cv2.waitKey(50) == 27:
        break

cv2.destroyAllWindows()

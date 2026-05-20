# save as ~/Downloads/verify_masks.py
import cv2, os, random
import numpy as np

DATASET_DIR = os.path.expanduser("~/Downloads/unet_dataset/")
SPLIT       = "train"   # change to "val" to check val set
N_SAMPLES   = 8         # how many random frames to check

# Colours per class (BGR)
CLASS_COLORS = {
    0: (0,   0,   0),    # background — black
    1: (0,   0,   255),  # line_1 — red
    2: (0,   255, 255),  # line_2 — yellow
    3: (0,   255, 0),    # line_3 — green
    4: (255, 255, 0),    # line_4 — cyan
    5: (255, 0,   0),    # line_5 — blue
}

img_dir  = os.path.join(DATASET_DIR, "images", SPLIT)
mask_dir = os.path.join(DATASET_DIR, "masks",  SPLIT)

files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
sample = random.sample(files, min(N_SAMPLES, len(files)))

out_dir = os.path.expanduser("~/Downloads/mask_verify/")
os.makedirs(out_dir, exist_ok=True)

for fname in sample:
    stem     = os.path.splitext(fname)[0]
    img      = cv2.imread(os.path.join(img_dir, fname))
    mask_path = os.path.join(mask_dir, stem + ".png")

    if not os.path.exists(mask_path):
        print(f"[SKIP] no mask for {fname}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Build colour overlay
    overlay = np.zeros_like(img)
    for cls_id, color in CLASS_COLORS.items():
        if cls_id == 0:
            continue
        overlay[mask == cls_id] = color

    # Blend with original (50/50)
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Add legend
    for cls_id, color in CLASS_COLORS.items():
        if cls_id == 0:
            continue
        label = f"line_{cls_id}"
        y = 20 + cls_id * 22
        cv2.rectangle(blended, (8, y-14), (24, y+2), color, -1)
        cv2.putText(blended, label, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Save
    out_path = os.path.join(out_dir, f"verify_{fname}")
    cv2.imwrite(out_path, blended)
    print(f"  ✅ {fname}  — unique mask classes: {np.unique(mask).tolist()}")

print(f"\nDone! Open ~/Downloads/mask_verify/ to inspect the overlays.")
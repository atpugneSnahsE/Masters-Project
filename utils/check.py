import cv2, os
import numpy as np
from collections import Counter

mask_dirs = [
    os.path.expanduser("~/Downloads/unet_dataset/masks/train"),
    os.path.expanduser("~/Downloads/unet_dataset/masks/val"),
]

all_classes = []
empty_masks = []

for mask_dir in mask_dirs:
    for fname in os.listdir(mask_dir):
        if not fname.endswith(".png"):
            continue
        mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
        uniq = np.unique(mask).tolist()
        all_classes.extend(uniq)
        if uniq == [0]:
            empty_masks.append(fname)

print("Class distribution across all masks:")
for cls, count in sorted(Counter(all_classes).items()):
    print(f"  class {cls}: {count} occurrences")

total = sum(1 for d in mask_dirs for f in os.listdir(d) if f.endswith(".png"))
print(f"\nTotal masks: {total}")
print(f"Empty masks (all background): {len(empty_masks)}")
if empty_masks[:5]:
    print(f"Examples: {empty_masks[:5]}")
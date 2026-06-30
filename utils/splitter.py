import os
import glob
import random
import numpy as np
import cv2
from tqdm import tqdm
import json

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
MASK_DIR = os.path.join(DATA_ROOT, "mask")
RGB_DIR = os.path.join(DATA_ROOT, "rgb")

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

CLASS_NAMES = {
    0: "CLASS_BG",
    1: "CLASS_EGO",
    2: "CLASS_LEFT_DASHED",
    3: "CLASS_LEFT_SOLID",
    4: "CLASS_RIGHT_DASHED",
    5: "CLASS_RIGHT_SOLID",
    6: "CLASS_OTHER",
    7: "CLASS_STOP",
    8: "CLASS_CROSS"
}

# Fix random seed for reproducibility
random.seed(42)

# ==========================================================

print("Scanning dataset files...")
mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))
print(f"Found {len(mask_files)} images total.")

# Extract the base names to match RGB and Masks perfectly
dataset_pairs = []
for mf in mask_files:
    base_name = os.path.basename(mf).replace(".png", "")
    rf = os.path.join(RGB_DIR, base_name + ".jpg")
    if os.path.exists(rf):
        dataset_pairs.append(base_name)

# Shuffle pairs
random.shuffle(dataset_pairs)

total_count = len(dataset_pairs)
train_end = int(total_count * SPLIT_RATIOS["train"])
val_end = train_end + int(total_count * SPLIT_RATIOS["val"])

splits = {
    "train": dataset_pairs[:train_end],
    "val": dataset_pairs[train_end:val_end],
    "test": dataset_pairs[val_end:]
}

print(f"\n--- Dataset Split Figures ---")
for split_name, items in splits.items():
    print(f"  {split_name.upper()}: {len(items)} samples ({len(items)/total_count*100:.1f}%)")

# Write split indexes out so your PyTorch DataLoader can easily read them
for split_name, items in splits.items():
    with open(os.path.join(DATA_ROOT, f"{split_name}.txt"), "w") as f:
        for item in items:
            f.write(f"{item}\n")
print(f"Generated train.txt, val.txt, and test.txt splits in {DATA_ROOT}!")

print("\nProfiling pixel-class distributions across the dataset... (Sampling 10% for speed)")
# Proportional pixel counting (Sampling 10% of masks is mathematically sufficient for a pixel distribution chart)
sample_rate = 0.10
sampled_masks = random.sample(mask_files, int(len(mask_files) * sample_rate))

global_pixel_counts = {c: 0 for c in CLASS_NAMES.keys()}

for mf in tqdm(sampled_masks, desc="Analyzing Mask Imbalances"):
    mask = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
    unique, counts = np.unique(mask, return_counts=True)
    for u, c in zip(unique, counts):
        if u in global_pixel_counts:
            global_pixel_counts[u] += int(c)

print("\n--- Raw Class Frequency / Pixel Imbalance Data ---")
total_pixels = sum(global_pixel_counts.values())

profile_summary = {}
for cid, name in CLASS_NAMES.items():
    p_count = global_pixel_counts[cid]
    p_percentage = (p_count / total_pixels) * 100 if total_pixels > 0 else 0
    profile_summary[name] = {"pixel_count": p_count, "percentage": round(p_percentage, 4)}
    print(f"  {name:<20}: {p_count:>12} px ({p_percentage:.4f}%)")

# Save stats to a JSON for quick plotting or pasting into your paper tables
with open(os.path.join(DATA_ROOT, "class_distribution_profile.json"), "w") as f:
    json.dump(profile_summary, f, indent=2)
print("\nSaved distribution profile data to 'class_distribution_profile.json'.")
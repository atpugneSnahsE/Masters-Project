"""Dataset diagnostics and analysis."""
import cv2, numpy as np, os
from pathlib import Path
from collections import defaultdict

def analyze_images(image_dir):
    """Analyze image statistics."""
    image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    
    print(f"Total images: {len(image_files)}")
    
    sizes = defaultdict(int)
    total_pixels = 0
    
    for img_file in image_files[:100]:  # Sample first 100
        img = cv2.imread(str(img_file))
        if img is not None:
            h, w = img.shape[:2]
            sizes[f"{w}x{h}"] += 1
            total_pixels += h * w
    
    print("\nImage sizes:")
    for size, count in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"  {size}: {count} images")
    
    print(f"\nAvg pixels per image: {total_pixels / min(100, len(image_files)):.0f}")

def analyze_masks(mask_dir):
    """Analyze mask statistics."""
    mask_files = sorted(Path(mask_dir).glob('*.png'))
    
    print(f"\nTotal masks: {len(mask_files)}")
    
    class_distribution = defaultdict(int)
    
    for mask_file in mask_files[:50]:  # Sample
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique, counts = np.unique(mask, return_counts=True)
            for cls_id, count in zip(unique, counts):
                class_distribution[cls_id] += count
    
    print("\nClass distribution:")
    for cls_id in sorted(class_distribution.keys()):
        count = class_distribution[cls_id]
        pct = 100 * count / sum(class_distribution.values())
        print(f"  Class {cls_id}: {pct:.1f}%")

if __name__ == '__main__':
    analyze_images('data/images/')
    analyze_masks('data/masks/')
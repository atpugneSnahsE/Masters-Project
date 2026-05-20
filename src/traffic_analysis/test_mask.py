"""Mask validation for traffic objects."""
import cv2, numpy as np, os
from pathlib import Path

def validate_mask(mask_path, img_path=None):
    """Check mask integrity and display."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Failed to load: {mask_path}")
        return False
    
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")
    print(f"Min: {mask.min()}, Max: {mask.max()}")
    
    if img_path and os.path.exists(img_path):
        img = cv2.imread(img_path)
        cv2.imshow('Original', img)
        cv2.imshow('Mask', mask * 50)  # scale for visibility
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return True

def validate_dataset(mask_dir):
    """Validate all masks in directory."""
    mask_files = sorted(Path(mask_dir).glob('*.png'))
    print(f"Found {len(mask_files)} masks")
    
    for mask_file in mask_files[:5]:  # Check first 5
        print(f"\nValidating: {mask_file.name}")
        validate_mask(str(mask_file))

if __name__ == '__main__':
    validate_dataset('data/masks/')
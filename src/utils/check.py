"""Data validation and checking utilities."""
import cv2, os, numpy as np
from pathlib import Path

def check_images(image_dir, verbose=True):
    """Validate all images in directory."""
    image_files = sorted(Path(image_dir).glob('*.jpg')) + sorted(Path(image_dir).glob('*.png'))
    print(f"Found {len(image_files)} images")
    
    valid_count = 0
    invalid_files = []
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            invalid_files.append(img_file.name)
            if verbose:
                print(f"  ✗ {img_file.name}")
        else:
            valid_count += 1
            if verbose and valid_count <= 5:
                print(f"  ✓ {img_file.name}: {img.shape}")
    
    print(f"\nValid: {valid_count}/{len(image_files)}")
    if invalid_files:
        print(f"Invalid files: {invalid_files}")
    
    return valid_count, invalid_files

def check_dataset(dataset_dir):
    """Check complete dataset structure."""
    print(f"Checking dataset: {dataset_dir}")
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if os.path.exists(split_dir):
            img_count = len(list(Path(split_dir).glob('*.jpg')))
            print(f"  {split}: {img_count} images")
        else:
            print(f"  {split}: NOT FOUND")

if __name__ == '__main__':
    check_images('data/images/')
    check_dataset('data/')
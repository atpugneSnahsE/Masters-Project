"""Camera calibration testing utilities."""
import cv2, numpy as np, os
from pathlib import Path

def load_calibration(calib_file):
    """Load camera calibration from file."""
    calib_data = np.load(calib_file)
    return {
        'K': calib_data['K'],           # Camera matrix
        'D': calib_data['D'],           # Distortion coefficients
        'rvec': calib_data['rvec'],     # Rotation vector
        'tvec': calib_data['tvec']      # Translation vector
    }

def test_calibration(image_path, calib_data):
    """Test calibration on image."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Undistort image
    K = calib_data['K']
    D = calib_data['D']
    undistorted = cv2.undistort(img, K, D)
    
    # Display comparison
    comparison = np.hstack([img, undistorted])
    cv2.imshow('Original vs Undistorted', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Calibration test complete")

def test_multiple(image_dir, calib_file):
    """Test calibration on multiple images."""
    calib_data = load_calibration(calib_file)
    image_files = sorted(Path(image_dir).glob('*.jpg'))[:5]
    
    for img_file in image_files:
        print(f"Testing: {img_file.name}")
        test_calibration(str(img_file), calib_data)

if __name__ == '__main__':
    calib_file = 'calibration.npz'
    image_dir = 'test_images/'
    test_multiple(image_dir, calib_file)
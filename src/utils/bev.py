"""Bird's Eye View (BEV) transformation."""
import cv2, numpy as np
from typing import Tuple

class BEVTransform:
    def __init__(self, src_points: np.ndarray, dst_points: np.ndarray, 
                 dst_size: Tuple[int, int] = (400, 800)):
        """Initialize BEV transformation.
        
        Args:
            src_points: Source points in original image (4 corners)
            dst_points: Destination points in BEV space
            dst_size: Output image size (height, width)
        """
        self.src_points = src_points.astype(np.float32)
        self.dst_points = dst_points.astype(np.float32)
        self.dst_size = dst_size
        
        # Compute perspective transform matrix
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective transform to image."""
        return cv2.warpPerspective(image, self.M, self.dst_size)
    
    def inverse_transform(self, bev_image: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Inverse perspective transform."""
        return cv2.warpPerspective(bev_image, self.M_inv, original_size)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D points to BEV space."""
        ones = np.ones((points.shape[0], 1))
        pts_homogeneous = np.hstack([points, ones])
        pts_transformed = pts_homogeneous @ self.M.T
        return (pts_transformed[:, :2] / pts_transformed[:, 2:3]).astype(int)

if __name__ == '__main__':
    # Example usage
    src = np.array([[100, 50], [300, 50], [350, 300], [50, 300]], dtype=np.float32)
    dst = np.array([[50, 0], [350, 0], [350, 800], [50, 800]], dtype=np.float32)
    
    bev = BEVTransform(src, dst, (800, 400))
    img = cv2.imread('frame.jpg')
    bev_img = bev.transform(img)
    cv2.imshow('BEV', bev_img)
    cv2.waitKey(0)
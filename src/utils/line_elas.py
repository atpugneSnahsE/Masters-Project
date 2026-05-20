"""Line elasticity calculation for lane detection."""
import numpy as np, cv2
from typing import Tuple

def calculate_line_elasticity(line_points: np.ndarray, window_size: int = 10) -> float:
    """Calculate elasticity of detected lane line.
    
    Elasticity measures how "wiggly" a line is.
    Lower = straighter, Higher = more curved.
    """
    if len(line_points) < window_size:
        return 0.0
    
    curvatures = []
    for i in range(1, len(line_points) - 1):
        p1 = line_points[i - 1]
        p2 = line_points[i]
        p3 = line_points[i + 1]
        
        # Calculate angle change (curvature)
        v1 = p2 - p1
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        curvatures.append(angle)
    
    return np.mean(curvatures) if curvatures else 0.0

def smooth_line(line_points: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth line using moving average."""
    if len(line_points) < kernel_size:
        return line_points
    
    smoothed = np.copy(line_points)
    for i in range(kernel_size // 2, len(line_points) - kernel_size // 2):
        smoothed[i] = np.mean(line_points[i - kernel_size//2:i + kernel_size//2 + 1], axis=0)
    
    return smoothed

if __name__ == '__main__':
    # Test with random line
    line = np.array([[i, np.sin(i/10) * 50 + 200] for i in range(100)])
    elasticity = calculate_line_elasticity(line)
    print(f"Elasticity: {elasticity:.4f}")
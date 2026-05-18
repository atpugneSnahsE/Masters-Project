# Utils Module

Utility functions for calibration, data collection, video processing, and diagnostics.

## Core Utilities
- `bev.py` - Bird's Eye View perspective transformation
- `bev_video.py` - BEV transformation for video streams
- `line_elas.py` - Lane line elasticity and smoothing

## Camera & Calibration
- `calib_test.py` - Camera calibration testing and validation
- `check.py` - Data validation utilities

## Data Processing
- `collect_data.py` - CARLA-based data collection
- `frames_extract.py` - Video frame extraction
- `diagnostics.py` - Dataset analysis and statistics

## Features

- Perspective transform for bird's eye view
- Video processing with BEV
- Camera calibration validation
- Data collection from CARLA simulator
- Frame extraction from videos
- Dataset statistics and diagnostics

## Usage

```python
from utils.bev import BEVTransform
import numpy as np

# Define corner points
src = np.array([[100, 50], [300, 50], [350, 300], [50, 300]], dtype=np.float32)
dst = np.array([[50, 0], [350, 0], [350, 800], [50, 800]], dtype=np.float32)

# Create BEV transformer
bev = BEVTransform(src, dst, (800, 400))
bev_image = bev.transform(image)
```

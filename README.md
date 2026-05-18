# Masters Project: Autonomous Vehicle Lane Detection & Traffic Analysis

A comprehensive autonomous vehicle perception system combining lane detection, traffic analysis, and object detection using deep learning.

## 📁 Project Structure

```
Masters-Project/
├── models/                    # Pre-trained and trained model weights
├── src/                       # Source code
│   ├── lane_detection/       # Lane detection models and inference
│   ├── traffic_analysis/     # Traffic and object detection
│   ├── utils/                # Utility functions and helpers
│   ├── data_processing/      # Data collection and preprocessing
│   └── visualization/        # Visualization and analysis tools
├── notebooks/                 # Jupyter notebooks for exploration
├── data/                      # Data files and datasets
├── output/                    # Generated outputs and results
└── docs/                      # Documentation
```

## 🚀 Key Components

### Lane Detection
- **UNet-based lane segmentation** using ResNet34 encoder
- **BEV (Bird's Eye View) transformation** for lane analysis
- **Sliding window detection** for robust lane finding
- **Polynomial fitting** for lane curvature calculation

### Traffic Analysis
- **YOLOv8 object detection** for traffic signals and vehicles
- **Lane marking classification** for road marking types
- **Ego-line classification** for vehicle position

### Data Processing
- **CARLA simulator integration** for synthetic data collection
- **Calibration utilities** for camera parameters
- **Frame extraction** and video processing

## 📊 Models Included

- `lane_model_final.pth` - Final lane detection model
- `best_lane_model.pth` - Best performing lane model
- `yolov8n.pt` - YOLOv8 Nano for object detection
- `yolov8n-oiv7.pt` - YOLOv8 with OIV7 dataset

## 🔧 Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download/prepare datasets
3. Configure CARLA simulator if using synthetic data
4. Run specific scripts from `src/` directory

## 📝 File Descriptions

See individual module README files for detailed documentation.

## 📚 References

- Advanced Lane Finding: https://medium.com/@mithi/advanced-lane-finding
- YOLOv8: https://docs.ultralytics.com/
- CARLA Simulator: https://carla.org/

---

**Status**: Active Development  
**Last Updated**: 2026-05-18
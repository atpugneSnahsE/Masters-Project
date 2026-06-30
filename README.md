# Masters Project: Autonomous Vehicle Lane Detection & Traffic Analysis

A comprehensive autonomous vehicle perception system combining lane detection, traffic analysis, and object detection using deep learning.

## Project Structure

```
Masters Thesis/
├── data_collection/          # CARLA data collection scripts
├── Datasets/                 # Training datasets (carla_lane_dataset, VIL100)
├── docs/                     # Documentation
├── evaluation/               # Evaluation metrics, analysis scripts
├── lane_pipelines/           # Lane detection inference pipelines
├── localization/             # Localization and RTK fusion
├── models/                   # Pre-trained and trained model weights
├── perception/               # Lane analysis, markings, ego classification
├── pipeline/                 # Integrated system pipelines
├── reports/                  # Generated reports and plots
├── sensor_experiments/       # Sensor simulation experiments
├── training/                 # Model training scripts
├── utils/                    # Utility functions (yolo, verification, etc.)
├── visualization/            # BEV, calibration, frame extraction
├── xai/                      # Explainable AI experiments
├── README.md
└── requirements.txt
```

## Key Components

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

## Models Included

- `models/carla_lane_models/lane_model_final.pth` - Final lane detection model (CARLA)
- `models/carla_lane_models/lane_model_best.pth` - Best performing lane model (CARLA)
- `models/vil100_model/final_model.pth` - VIL100-trained lane model
- `models/vil100_model/best_model.pth` - Best VIL100 lane model
- `models/yolov8n.pt` - YOLOv8 Nano for object detection
- `models/yolov8n-oiv7.pt` - YOLOv8 with OIV7 dataset
- `models/traffic_best.pt` - Best traffic detection model

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Download/prepare datasets
3. Configure CARLA simulator if using synthetic data
4. Run specific scripts from respective directories

## References

- Advanced Lane Finding: https://medium.com/@mithi/advanced-lane-finding
- YOLOv8: https://docs.ultralytics.com/
- CARLA Simulator: https://carla.org/

---

**Status**: Active Development  
**Last Updated**: 2026-06-30

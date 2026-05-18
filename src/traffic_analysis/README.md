# Traffic Analysis Module

Traffic detection and vehicle classification using YOLO and object detection models.

## Files

### Core Models
- `yolo.py` - YOLO model wrapper with video processing
- `combined.py` - Multi-model traffic detection pipeline
- `combined2.py` - Enhanced multi-model detector with metrics

### Testing
- `test_model_1.py` - Single image inference test
- `test_mask.py` - Mask validation and dataset checking

## Features

- YOLOv8 Nano for real-time detection
- Support for vehicle and traffic signal detection
- Video processing with configurable confidence threshold
- Multi-model ensemble detection
- Mask validation utilities

## Usage

```python
from traffic_analysis.yolo import TrafficYOLO

# Load model
detector = TrafficYOLO(model_name='yolov8n.pt', conf=0.5)

# Process video
detector.process_video('input.mp4', 'output.mp4')

# Single frame detection
detections = detector.detect(frame)
```

## Models

- **yolov8n.pt** - Nano model (6 MB) - Real-time, CPU capable
- **yolov8m.pt** - Medium model - Better accuracy, higher latency
- **Custom models** - Fine-tuned on traffic datasets

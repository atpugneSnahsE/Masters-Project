# Visualization Module

Visualization and explainability tools for analysis and debugging.

## Files

### Lane Visualization
- `video_lane_pipeline.py` - Complete video lane detection pipeline
- `video_lane_pipeline_v2.py` - Improved pipeline version

### Explainability & Analysis
- `xai.py` - Explainable AI analysis (version 1)
- `xai2.py` - Explainable AI analysis (version 2)

## Usage

```python
# Visualize lane detection results
from src.visualization.video_lane_pipeline import process_video
process_video('input.mp4', 'output.mp4')
```

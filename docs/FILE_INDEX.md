# Complete File Index

## Source Code (src/)

### Lane Detection (src/lane_detection/)
| File | Purpose |
|------|----------|
| `train_lane_model.py` | Train UNet lane segmentation model with ResNet34 encoder |
| `train_lane_detection.py` | Alternative lane detection training approach |
| `train_lanenet.py` | Train LaneNet architecture for lane detection |
| `train_vil100.py` | Train on VIL100 dataset |
| `finetune_vil100.py` | Fine-tune pre-trained model on VIL100 |
| `run_lane_model.py` | Inference: Run lane detection on CARLA/video data |
| `run_lane_markings.py` | Detect and classify lane marking types |
| `run_lane_markings_night.py` | Lane marking detection optimized for night conditions |
| `analyze_lanes.py` | Analyze detected lanes: curvature, position, quality metrics |
| `lane_marking_classifier.py` | Classify road marking types (solid, dashed, etc.) |
| `ego_line_classifier.py` | Classify vehicle position relative to detected lanes |
| `verify_masks.py` | Validate segmentation mask quality and correctness |

### Traffic Analysis (src/traffic_analysis/)
| File | Purpose |
|------|----------|
| `yolo.py` | YOLOv8 object detection for traffic signals, vehicles, pedestrians |
| `combined.py` | Combined lane detection + traffic analysis pipeline |
| `combined2.py` | Alternative combined pipeline approach |
| `test_model_1.py` | Test and validate traffic detection models |
| `test_mask.py` | Test segmentation masks from detection models |

### Utilities (src/utils/)
| File | Purpose |
|------|----------|
| `collect_data.py` | Collect training data from CARLA simulator |
| `frames_extract.py` | Extract individual frames from video files |
| `bev.py` | Bird's Eye View transformation utilities |
| `bev_video.py` | BEV visualization for video sequences |
| `line_elas.py` | Elastic line analysis and fitting |
| `diagnostics.py` | System diagnostics and debugging tools |
| `check.py` | Validation and integrity checking utilities |
| `calib_test.py` | Camera calibration and validation |

### Data Processing (src/data_processing/)
| File | Purpose |
|------|----------|
| `test.py` | General testing utilities and test cases |
| `test2.py` | Additional test functions |

### Visualization (src/visualization/)
| File | Purpose |
|------|----------|
| `video_lane_pipeline.py` | Complete video lane detection pipeline with visualization |
| `video_lane_pipeline_v2.py` | Improved lane detection video pipeline (v2) |
| `xai.py` | Explainable AI analysis tools (version 1) |
| `xai2.py` | Explainable AI analysis tools (version 2) |

## Configuration Files
| File | Purpose |
|------|----------|
| `.gitignore` | Git ignore patterns (data, models, cache) |
| `requirements.txt` | Python package dependencies |
| `README.md` | Main project documentation |

---

**Last Updated**: 2026-05-18  
**Total Python Scripts**: 31  
**Model Checkpoints**: 6  

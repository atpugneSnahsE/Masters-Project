# Complete File Index

## Data Collection (data_collection/)
| File | Purpose |
|------|---------|
| `carla_data_collection.py` | Collect training data from CARLA simulator |
| `collect_data.py` | General data collection utilities |
| `manual_control.py` | Manual control interface for data collection |

## Lane Pipelines (lane_pipelines/)
| File | Purpose |
|------|---------|
| `combined.py` | Combined lane detection + traffic analysis pipeline |
| `combined2.py` | Alternative combined pipeline approach |
| `night_drive.py` | Night-time lane detection pipeline |
| `run_lane_markings.py` | Detect and classify lane marking types |
| `run_lane_markings_night.py` | Lane marking detection optimized for night conditions |
| `run_lane_model.py` | Inference: Run lane detection on CARLA/video data |
| `run_model.py` | General model inference runner |
| `video_lane_pipeline.py` | Complete video lane detection pipeline with visualization |
| `video_lane_pipeline_v2.py` | Improved lane detection video pipeline (v2) |

## Localization (localization/)
| File | Purpose |
|------|---------|
| `carla_localization_fusion.py` | CARLA localization with sensor fusion |
| `localization.py` | Localization utilities |
| `rtk_localization.py` | RTK-GPS localization |

## Perception (perception/)
| File | Purpose |
|------|---------|
| `analyze_lanes.py` | Analyze detected lanes: curvature, position, quality metrics |
| `ego_line_classifier.py` | Classify vehicle position relative to detected lanes |
| `lane_marking_classifier.py` | Classify road marking types (solid, dashed, etc.) |
| `line_elas.py` | Elastic line analysis and fitting |
| `perception.py` | Main perception module |

## Pipeline (pipeline/)
| File | Purpose |
|------|---------|
| `config.py` | System configuration parameters |
| `extended_pipeline.py` | Extended system pipeline |
| `final_system_pipeline.py` | Final integrated system pipeline |
| `main.py` | Main entry point for the system |

## Sensor Experiments (sensor_experiments/)
| File | Purpose |
|------|---------|
| `sensor_exp1.py` | Sensor experiment 1 |
| `sensor_exp1_montecarlo.py` | Monte Carlo validation for experiment 1 |
| `sensor_exp2.py` | Sensor experiment 2 |
| `sensor_exp2_montecarlo.py` | Monte Carlo validation for experiment 2 |
| `sensor_exp3.py` | Sensor experiment 3 |
| `sensor_exp4.py` | Sensor experiment 4 |

## Training (training/)
| File | Purpose |
|------|---------|
| `carla_train.py` | Train lane detection on CARLA data |
| `finetune_vil100.py` | Fine-tune pre-trained model on VIL100 |
| `train_lane_detection.py` | Lane detection training approach |
| `train_lane_model.py` | Train UNet lane segmentation model with ResNet34 encoder |
| `train_lanenet.py` | Train LaneNet architecture for lane detection |
| `train_plots.py` | Training visualization plots |
| `train_plots2.py` | Additional training plots |
| `train_vil100.py` | Train on VIL100 dataset |

## Utils (utils/)
| File | Purpose |
|------|---------|
| `check.py` | Validation and integrity checking utilities |
| `map_probe.py` | Map probing utilities |
| `splitter.py` | Data splitting utilities |
| `test_mask.py` | Test segmentation masks from detection models |
| `verify_masks.py` | Validate segmentation mask quality and correctness |
| `yolo.py` | YOLOv8 object detection for traffic signals, vehicles, pedestrians |

## Visualization (visualization/)
| File | Purpose |
|------|---------|
| `bev.py` | Bird's Eye View transformation utilities |
| `bev_video.py` | BEV visualization for video sequences |
| `calib_test.py` | Camera calibration and validation |
| `frames_extract.py` | Extract individual frames from video files |
| `plot.py` | General plotting utilities |
| `visualization.py` | Visualization tools and helpers |

## XAI (xai/)
| File | Purpose |
|------|---------|
| `xai.py` | Explainable AI analysis tools |
| `experiment271.py` | XAI experiment 271 |
| `experiment272.py` | XAI experiment 272 |
| `experiment273.py` | XAI experiment 273 |
| `experiment274.py` | XAI experiment 274 |
| `experiment275.py` | XAI experiment 275 |

## Evaluation (evaluation/)
| File | Purpose |
|------|---------|
| `comparison.py` | Model comparison and analysis |
| `diagnostics.py` | System diagnostics and debugging |
| `extract_thesis_metrics.py` | Extract metrics for thesis reporting |
| `sim_comparison.py` | Simulation comparison analysis |
| `simulation_metrics_comp.py` | Simulation metrics comparison |
| `test_model_1.py` | Test and validate traffic detection models |

## Model Checkpoints (models/)
| File | Purpose |
|------|---------|
| `carla_lane_models/lane_model_final.pth` | Final lane detection model (CARLA) |
| `carla_lane_models/lane_model_best.pth` | Best performing lane model (CARLA) |
| `vil100_model/final_model.pth` | VIL100-trained lane model |
| `vil100_model/best_model.pth` | Best VIL100 lane model |
| `yolov8n.pt` | YOLOv8 Nano for object detection |
| `yolov8n-oiv7.pt` | YOLOv8 with OIV7 dataset |
| `traffic_best.pt` | Best traffic detection model |

## Reports (reports/)
| Directory | Purpose |
|-----------|---------|
| `day_vs_night_comparison/` | Day vs night performance comparison reports |
| `segmentation_plots/` | Segmentation visualization plots |
| `system_performance/` | System performance analysis reports |

## Datasets (Datasets/)
| Directory | Purpose |
|-----------|---------|
| `carla_lane_dataset/` | CARLA simulator lane detection dataset |
| `VIL100/` | VIL100 lane detection dataset |

## Configuration Files
| File | Purpose |
|------|---------|
| `requirements.txt` | Python package dependencies |
| `README.md` | Main project documentation |

---

**Last Updated**: 2026-06-30  
**Total Python Scripts**: 62  
**Model Checkpoints**: 7

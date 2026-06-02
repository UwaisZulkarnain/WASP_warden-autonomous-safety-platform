# Computer Vision Module

This folder contains the YOLOv8-based PPE (Personal Protective Equipment) detection module developed for the WASP (Warden Autonomous Safety Platform) project.

The module is designed to monitor construction site safety by detecting workers and verifying compliance with required PPE equipment in real-time camera feeds.

## Detected Classes

| Class ID | Class Name |
| -------- | ---------- |
| 0        | helmet     |
| 1        | gloves     |
| 2        | vest       |
| 3        | boots      |
| 4        | goggles    |
| 5        | none       |
| 6        | Person     |
| 7        | no_helmet  |
| 8        | no_goggle  |
| 9        | no_gloves  |
| 10       | no_boots   |

## Files Description

| File                  | Description                                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `best_construction_ppe.pt`             | Final trained YOLOv8 model used for deployment and real-time inference.                                          |
| `testyolo.py`         | Script for testing the trained model using a webcam or video source.                                             |
| `Training_yolov8n.ipynb` | Jupyter notebook used for model training and experimentation.                                                    |
| `data.yaml`           | Dataset configuration file containing class definitions and dataset paths.                                       |
| `requirements.txt`    | Python dependencies required to run the module.                                                                  |

## Model Information

* Model Architecture: YOLOv8n
* Task: Object Detection
* Framework: Ultralytics YOLO
* Dataset: Construction-PPE Dataset
* Purpose: Real-time PPE compliance monitoring for construction site safety

## Running the Detection System

Install dependencies:

```bash
pip install -r requirements.txt
```

Run webcam detection:

```bash
python testyolo.py
```

## Project Workflow

```text
Camera Feed
      ↓
YOLOv8 PPE Detection
      ↓
PPE Compliance Analysis
      ↓
Violation Detection
      ↓
Alert / Dashboard Integration
```

## Notes

* Use `best.pt` for deployment and testing.
* The training dataset is not included in this repository due to storage limitations.
* This module is intended to be integrated with the WASP IoT monitoring and dashboard components.
* The model can be connected to ESP32 camera streams, CCTV feeds, or local webcams for real-time monitoring.

# Computer Vision Module

This folder contains the YOLOv8-based PPE (Personal Protective Equipment) detection module developed for the WASP (Warden Autonomous Safety Platform) project.

The model is designed to detect workers and identify whether safety helmets are being worn in real-time camera feeds.

## Detected Classes

| Class ID | Class Name |
| -------- | ---------- |
| 0        | helmet     |
| 1        | head       |
| 2        | person     |

## Files Description

| File                  | Description                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `best.pt`             | Final trained YOLOv8 model used for deployment and inference.                                                     |
| `testyolo.py`         | Script for testing the trained model using a webcam or video source.                                              |
| `training_yolo.ipynb` | Jupyter notebook used for model training and experimentation.                                                     |
| `data.yaml`           | Dataset configuration file containing class names and dataset paths.                                              |
| `requirements.txt`    | Python dependencies required to run the project.                                                                  |

## Model Information

* Model Architecture: YOLOv8n
* Task: Object Detection
* Framework: Ultralytics YOLO
* Training Purpose: PPE Detection for Construction Site Safety Monitoring

## Running the Detection System

Install dependencies:

```bash
pip install -r requirements.txt
```

Run webcam detection:

```bash
python testyolo.py
```

## Notes

* Use `best.pt` for deployment and testing.
* The training dataset is not included in this repository due to storage limitations.
* This module is intended to be integrated with the WASP IoT and dashboard components.

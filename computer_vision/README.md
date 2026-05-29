# Computer Vision Module

This folder contains the YOLOv8 PPE detection module for the WASP (Warden Autonomous Safety Platform) project.

## Model

The model was trained using YOLOv8n for real-time PPE detection.

### Detected Classes

* helmet
* head
* person

### Main Model File

```text
best.pt
```

## Project Structure

```text
computer_vision/
├── README.md
├── best.pt
├── data.yaml
├── requirements.txt
├── testyolo.py
└── training_yolo.ipynb
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Detection

```bash
python testyolo.py
```

## Notes

* Model: YOLOv8n
* Framework: Ultralytics YOLO
* Purpose: PPE detection for construction site safety monitoring
* The training dataset is not included in this repository due to file size limitations.

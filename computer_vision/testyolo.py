from ultralytics import YOLO

model = YOLO(
    r"best.pt"
)

results = model.predict(
    source=0,
    conf=0.5,
    show=True
)
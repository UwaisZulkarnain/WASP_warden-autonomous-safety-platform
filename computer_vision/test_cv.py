"""Born: Use this to test your YOLOv8 model independently"""
from ultralytics import YOLO
import cv2

MODEL_PATH = "best.pt"  # Change if your model has a different name

def main():
    print("[TEST] Loading model...")
    model = YOLO(MODEL_PATH)
    print(f"[TEST] Model loaded. Classes: {model.names}")

    cap = cv2.VideoCapture(0)  # Laptop webcam
    if not cap.isOpened():
        print("[TEST] ERROR: Cannot open webcam")
        return

    print("[TEST] Press 'Q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)
        annotated = results[0].plot()

        # Print detections
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"[DETECTED] {model.names[cls]} ({conf:.2f})")

        cv2.imshow("WASP CV Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TEST] Done")

if __name__ == "__main__":
    main()

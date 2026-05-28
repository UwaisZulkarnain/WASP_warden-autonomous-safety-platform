import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO
import json
import threading
from flask import Flask, Response

STREAM_URL   = "http://192.168.100.178/stream"
MQTT_BROKER  = "localhost"                
MQTT_PORT    = 1883
TOPIC_RESULT = "esp32/cv/detections"


app = Flask(__name__)
output_frame = None
lock = threading.Lock()


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to MQTT broker!")
    else:
        print(f"Failed to connect, reason code: {reason_code}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()
except Exception as e:
    print(f"MQTT connection error: {e}")
    exit()

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("YOLO model loaded!")


def process_stream():
    global output_frame

    print(f"Connecting to ESP32 stream: {STREAM_URL}")
    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Failed to open stream!")
        return

    print("Stream opened successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost, retrying...")
            cap.release()
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        detections = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls)]
                conf  = float(box.conf)
                detections.append({
                    "label": label,
                    "confidence": round(conf * 100, 1)
                })


        payload = json.dumps({
            "count":      len(detections),
            "detections": detections
        })
        client.publish(TOPIC_RESULT, payload)
        print(f"Published: {payload}")


        with lock:
            output_frame = annotated.copy()


def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            _, buffer = cv2.imencode('.jpg', output_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream')
def stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '<h1>CV Proxy Running</h1><img src="/stream" width="640"/>'


if __name__ == '__main__':

    t = threading.Thread(target=process_stream)
    t.daemon = True
    t.start()


    print("Re-stream available at: http://localhost:5000/stream")
    print("Or from other devices: http://192.168.X.ZZZ:5000/stream")
    app.run(host='0.0.0.0', port=5000, threaded=True)
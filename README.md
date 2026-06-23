# WASP - Warden Autonomous Safety Platform

<div align="center">

**UTM FAI Showcase 2026**

*Team: Uwais (Backend/Integration), Paen (IoT), Born (CV)*

</div>

---

## :warning: Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.10 | Recommended for best compatibility |
| pip | latest | `python -m pip install --upgrade pip` |
| Windows/Mac/Linux | any | Cross-platform supported |
| Node-RED | latest | For the legacy dashboard (optional) |
| Ollama | latest | Optional, for local LLM agent mode |

---

## :rocket: Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/UwaisZulkarnain/WASP_warden-autonomous-safety-platform.git
cd WASP_warden-autonomous-safety-platform

# 2. Create and activate virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Verify environment
python check_env.py

# 5. Generate TTS warning audio files (requires internet)
python tts/generate_tts.py

# 6. Configure settings in wasp_backend.py (see Configuration section)

# 7. Run the backend
python wasp_backend.py

# 8. Open dashboard in browser
# http://localhost:5000
```

---

## :file_folder: Project Structure

```
WASP_warden-autonomous-safety-platform/
├── wasp_backend.py              # Main Flask backend (4600+ lines)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── check_env.py                 # Environment verification script
├── train_model.md               # ML model training notebook export
│
├── cv_final.pt                  # Trained YOLOv8 PPE detection model
├── computer_vision/
│   ├── best_construction_ppe.pt # Alternative YOLO model
│   ├── data.yaml                # YOLO dataset config
│   ├── README.md                # CV-specific documentation
│   ├── requirements.txt         # CV-specific deps
│   ├── test_cv.py               # CV testing script
│   ├── testyolo.py              # YOLO quick test
│   └── Training_yolov8n.ipynb   # YOLO training notebook
│
├── camera/
│   ├── camera_setup_esp32.cpp   # ESP32-S3 camera firmware
│   ├── cv_test_esp.py           # ESP32 stream client
│   └── esp32_sensors.ino        # ESP32 sensor sketches
│
├── esp32/
│   ├── main.cpp                 # ESP32 main firmware
│   ├── mqtt_reciever.py         # ESP32 MQTT subscriber
│   └── model/
│       ├── model_analog.pkl     # ML anomaly detection model
│       ├── model_env.pkl
│       ├── scaler_analog.pkl
│       ├── scaler_env.pkl
│       └── threshold.npy
│
├── iot_ml_model/
│   ├── train_model.ipynb        # IsolationForest training
│   └── dataset/
│       ├── kl_weather_openmeteo.csv
│       └── sensor_data_extended.csv
│
├── tts/
│   └── generate_tts.py          # Generate Bahasa Malaysia TTS audio
├── warnings/                    # Pre-generated MP3 warning files (generated)
├── logs/                        # Runtime log storage
├── reports/                     # Daily safety reports (JSON + TXT)
├── dashboard/
│   └── flows.json               # Node-RED dashboard flows
│
├── backend_stdout.log           # Backend stdout log
├── backend_stderr.log           # Backend stderr log
└── wasp.db                      # SQLite database (auto-created)
```

---

## :gear: Configuration

### Configuration is done by editing `wasp_backend.py` directly. Key settings:

### Camera Source

```python
USE_XIAO = False              # True = ESP32 camera stream, False = webcam
XIAO_IP = "192.168.4.1"       # ESP32 IP when USE_XIAO = True
XIAO_STREAM_URL = f"http://{XIAO_IP}:81/stream"
```

### ESP32 Serial Connection

```python
ESP32_PORT = "COM3"           # Windows: "COM3", Mac: "/dev/ttyUSB0", Linux: "/dev/ttyUSB0"
ESP32_BAUD = 115200
```

### YOLO Model

```python
MODEL_PATH = "cv_final.pt"    # Path to trained PPE detection model
SIMULATE_CV = False           # True = fake detection data for demo
CONFIDENCE_THRESHOLD = 0.35   # Minimum confidence for detections
```

### Agent & AI Settings

```python
AGENT_MODE = "groq"           # "ollama" (local) or "groq" (cloud)

# Ollama (local LLM)
OLLAMA_URL = "http://192.168.212.193:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 60

# Groq (cloud LLM fallback)
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = "your_groq_api_key_here"
```

### MQTT Broker

```python
BROKER_HOST = "192.168.100.218"   # or "localhost" if same device
BROKER_PORT = 1883
TOPIC = "sensors/#"
USERNAME = ""
PASSWORD = ""
```

### Thresholds

```python
WARNING_COOLDOWN = 30          # Seconds between HIGH/CRITICAL alerts
MEDIUM_COOLDOWN = 60           # Seconds between MEDIUM alerts
HEAT_THRESHOLD = 38.0          # Celsius - trigger heat warnings above this
ML_DEMO_THRESHOLD = -0.6500    # IsolationForest anomaly threshold
```

---

## :desktop_computer: System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WASP Backend (Flask)                         │
│                         wasp_backend.py                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ CV Thread    │    │ Sensor Thread│    │ Daily Report Thread  │  │
│  │ (YOLOv8)     │    │ (Serial/MQTT)│    │ (18:00 daily)        │  │
│  │              │    │              │    │                      │  │
│  │ - Person     │    │ - Temp       │    │ - JSON report        │  │
│  │   detection  │    │ - Humidity   │    │ - Narrative report   │  │
│  │ - PPE check  │    │ - Motion     │    │   (Groq-generated)   │  │
│  │ - Spatial    │    │ - IR/Obstacle│    │                      │  │
│  │   matching   │    │ - Sound/Mic  │    │                      │  │
│  │              │    │ - Air Quality│    │                      │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
│         │                   │                       │               │
│         └───────────────────┼───────────────────────┘               │
│                             │                                       │
│                    ┌────────▼────────┐                              │
│                    │  Agent Engine   │                              │
│                    │  (WASPAgent)    │                              │
│                    │                 │                              │
│                    │  - Rule-based   │                              │
│                    │  - Ollama/Groq  │                              │
│                    │  - Tool calling │                              │
│                    │  - ML anomaly   │                              │
│                    │    detection    │                              │
│                    └────────┬────────┘                              │
│                             │                                       │
│  ┌──────────┐   ┌──────────┴──────────┐    ┌───────────────────┐  │
│  │ Database │   │     TTS Engine       │    │  MQTT Publisher   │  │
│  │ (SQLite) │   │  (pygame + gTTS)     │    │  → ESP32 alerts   │  │
│  │          │   │                      │    │                    │  │
│  │ - alerts │   │ - Pre-rendered MP3   │    │                    │  │
│  │ - agent_ │   │   warnings (BM/EN)   │    │                    │  │
│  │   decisions│  │                      │    │                    │  │
│  └──────────┘   └──────────────────────┘    └───────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Flask Web Dashboard                        │  │
│  │              http://localhost:5000                            │  │
│  │                                                              │  │
│  │  • Live camera feed with overlay                            │  │
│  │  • Environmental sensors (temp, humidity, gas, motion)      │  │
│  │  • PPE status (helmet, vest, goggles, gloves, boots)        │  │
│  │  • AI Agent decisions & reasoning                           │  │
│  │  • Per-person detection cards                               │  │
│  │  • Alert history table                                      │  │
│  │  • Ollama/Groq mode toggle                                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
               ┌────────────────┼────────────────┐
               │                │                │
      ┌────────▼────────┐ ┌────▼─────┐ ┌────────▼────────┐
      │   ESP32-S3      │ │  MQTT    │ │  Node-RED        │
      │   Camera        │ │ Broker   │ │  Dashboard       │
      │   (XIAO)        │ │          │ │  (legacy)        │
      │                 │ │ 1883     │ │                  │
      │ - Video stream  │ │          │ │ - Custom panels  │
      │ - Sensors       │ │          │ │ - Flow-based     │
      │ - Buzzer/LED    │ │          │ │   logic          │
      └─────────────────┘ └──────────┘ └─────────────────┘
```

---

## :mag: Features

### :camera: Computer Vision
- **YOLOv8** PPE detection (helmet, vest, goggles, gloves, boots)
- **Per-person spatial matching** — PPE items matched to individual bounding boxes
- **Segment-aware verification** — adapts checks based on how much of the person is visible
- **GPU acceleration** — CUDA auto-detection with fallback to CPU
- **Real-time FPS counter** and GPU utilization overlay

### :robot: AI Safety Agent
- **Dual LLM support**: Ollama (local) + Groq (cloud fallback)
- **Tool calling / Function calling** — agent can:
  - Query sensor trends (`get_sensor_trend`)
  - Check violation history (`get_violation_history`)
  - Get worker count (`get_worker_count`)
  - Trigger ESP32 alerts (`trigger_esp32_alert`)
  - Push supervisor notifications (`push_supervisor_alert`)
  - Log incidents (`log_incident`)
- **Rule-based fallback** when LLMs unavailable
- **Severity tiers**: LOW → MEDIUM → HIGH → CRITICAL
- **Cooldown logic** to prevent alert spam

### :chart_with_upwards_trend: ML Anomaly Detection
- **IsolationForest** for environmental anomaly detection
- Features: temperature, humidity, sound (mic), air quality (MQ2)
- **Separate models** for environmental and analog sensor data
- **LIME explanations** for interpretable anomaly detection
- Configurable sensitivity threshold

### :satellite: IoT & Sensors (ESP32)
- **MQTT communication** over WiFi
- **DHT11** temperature & humidity
- **HC-SR501** PIR motion detection
- **IR obstacle** sensor
- **MQ2** gas/air quality
- **Microphone** sound level
- **Buzzer + LED** local alarms via MQTT commands

### :sound: Audio / TTS
- **Pre-generated Bahasa Malaysia** warning messages via gTTS
- **Pygame** mixer for async playback
- 13 warning variants covering all PPE combinations + heat alerts

### :bar_chart: Reporting
- **Daily safety reports** at 18:00 (6 PM)
- **JSON** raw data export (alerts, decisions, sensor peaks)
- **Narrative TXT** — Groq-generated OSHA-style English report
- Automatic suppression if report already generated

### :floppy_disk: Database
- **SQLite** with two tables:
  - `alerts` — timestamped safety alerts
  - `agent_decisions` — full agent context + decision logs

---

## :wrench: Hardware Requirements

| Component | Purpose | Notes |
|-----------|---------|-------|
| Laptop/PC | Backend server | Runs Flask, YOLO, AI agent |
| Webcam | Person + PPE detection | USB camera, 640×480 minimum |
| ESP32-S3 (XIAO) | Optional camera source | Seeed XIAO ESP32S3 + OV5647 camera |
| ESP32 | Sensors & actuators | DHT11, MQ2, HC-SR501, buzzer, LED |
| Jumper wires, breadboard | Wiring | Standard 3.3V/5V logic |

### Wiring (ESP32 Sensors)

| Sensor | ESP32 Pin | Notes |
|--------|-----------|-------|
| DHT11 Data | GPIO4 | Pull-up resistor 10kΩ |
| MQ2 Analog | GPIO34 | ADC1 pin |
| HC-SR501 OUT | GPIO18 | Digital input |
| IR Obstacle | GPIO19 | Digital input |
| Buzzer | GPIO21 | Active or passive |
| LED | GPIO22 | With current-limiting resistor |

---

## :hammer_and_wrench: ESP32 Setup

### Camera (Seeed XIAO ESP32S3)

1. Install **PlatformIO** in VS Code
2. Configure `platformio.ini`:

```ini
[env:seeed_xiao_esp32s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200

lib_deps =
  espressif/esp32-camera @ ^2.0.0
```

3. Copy `camera/camera_setup_esp32.cpp` to `main.cpp`
4. Upload and note the IP address printed in Serial Monitor

### Sensors

1. Open `esp32/main.cpp` and configure WiFi credentials
2. Upload via PlatformIO or Arduino IDE
3. Open Serial Monitor (115200 baud) to confirm sensor readings

---

## :computer: Dashboard

### Flask Dashboard (Primary)

The Flask web dashboard is served automatically when running `wasp_backend.py`. It provides:

- Live annotated video feed with PPE bounding boxes
- Real-time environmental sensor displays
- Per-person PPE status cards
- AI Agent reasoning display with tool call history
- Alert history with color-coded severity
- Ollama/Groq mode toggle buttons

Access at: **`http://localhost:5000`**

### Node-RED Dashboard (Legacy)

1. Import `dashboard/flows.json` into Node-RED
2. Ensure **MQTT broker IP** matches the IP running `cv_test_esp.py`
3. Edit the template node `img src` to match the ESP32 camera IP
4. Access at: **`http://<node-red-ip>:1880/dashboard`**

---

## :test_tube: Training the Models

### YOLOv8 PPE Detection

1. Prepare dataset in YOLO format (images + labels)
2. Configure `computer_vision/data.yaml`
3. Train:

```bash
cd computer_vision
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=416
```

4. Copy best weights to root as `cv_final.pt`

### IsolationForest Anomaly Detection

1. Collect sensor data into CSV format
2. Run `iot_ml_model/train_model.ipynb`
3. Models are saved to `esp32/model/`:
   - `model_analog.pkl`, `scaler_analog.pkl`
   - `model_env.pkl`, `scaler_env.pkl`
   - `threshold.npy`

---

## :book: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard page |
| `/video_feed` | GET | MJPEG video stream |
| `/api/sensors` | GET | JSON: current sensor readings |
| `/api/cv` | GET | JSON: CV detection results |
| `/api/agent` | GET | JSON: latest agent decision |

### Example API Response

```json
// GET /api/sensors
{
  "temperature": 32.5,
  "humidity": 65.0,
  "motion": 1,
  "ir": 0,
  "sound": 0,
  "air_quality": 400,
  "last_update": "14:32:05"
}

// GET /api/cv
{
  "person_count": 2,
  "persons": [...],
  "any_violation": true,
  "global_helmet": false,
  "global_vest": true,
  "global_goggles": true,
  "global_gloves": false,
  "global_boots": true,
  "last_update": "14:32:05"
}

// GET /api/agent
{
  "risk_level": "HIGH",
  "reasoning": "Critical PPE missing: helmet.",
  "speak_bm": "Perhatian! Helmet tidak dipakai!",
  "speak_en": "Warning! Helmet not worn!",
  "notify_supervisor": true,
  "action_tier": 3,
  "model_used": "groq",
  "tool_calls": [...],
  "response_time_ms": 1240
}
```

---

## :pencil: Database Schema

### `alerts` table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | ISO datetime |
| alert_type | TEXT | e.g. `AGENT_VIOLATION`, `PPE_MISSING` |
| details | TEXT | Alert description |
| status | TEXT | `ACTIVE` or `RESOLVED` |

### `agent_decisions` table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Auto-increment |
| timestamp | TEXT | ISO datetime |
| context_json | TEXT | Full site context at decision time |
| decision_json | TEXT | Agent decision payload |
| risk_level | TEXT | LOW/MEDIUM/HIGH/CRITICAL |
| model_used | TEXT | groq/ollama/rule-based |
| tool_calls | TEXT | JSON array of tool invocations |
| response_time_ms | INTEGER | LLM response latency |

---

## :bug: Troubleshooting

### Camera not opening
- Check USB camera is not being used by another application
- For ESP32: confirm IP and stream URL are correct
- Try changing `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` to `320`

### Serial port error
- Windows: Use Device Manager to find correct COM port
- Mac/Linux: Use `ls /dev/tty.*` or `ls /dev/ttyUSB*`
- Ensure no other program (e.g. Arduino Serial Monitor) is using the port

### MQTT connection failed
- Verify broker is running: `mosquitto_sub -h localhost -t "#"`
- Check firewall allows port 1883
- Set `USERNAME` and `PASSWORD` if broker requires auth

### Groq API errors
- Verify `GROQ_API_KEY` is valid
- Check internet connectivity
- API rate limits: free tier has ~30 requests/minute
- Fallback to `AGENT_MODE = "rule-based"` if needed

### Ollama connection failed
- Ensure Ollama is running: `ollama serve`
- Pull the model: `ollama pull llama3.2:3b`
- Check `OLLAMA_URL` is reachable from your machine

### No TTS audio
- Run `python tts/generate_tts.py` to generate MP3 files
- Check `warnings/` directory exists and contains `.mp3` files
- On Linux: install `sudo apt install python3-pygame` (SDL dependencies)

### Model file not found
- Copy the trained YOLO model (`cv_final.pt`) to the project root
- For ML models: ensure `esp32/model/` contains all `.pkl` and `.npy` files

### GPU not detected
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Install PyTorch with CUDA: https://pytorch.org/get-started/locally/
- Backend automatically falls back to CPU

---

## :white_check_mark: Pre-Run Checklist

Run `python check_env.py` before the showcase. It verifies:

- [ ] Python 3.10+ installed
- [ ] All required modules importable
- [ ] YOLO model file present (`cv_final.pt`)
- [ ] TTS audio files generated in `warnings/`
- [ ] ESP32 connected and configured
- [ ] MQTT broker running
- [ ] Groq API key configured (or Ollama running)
- [ ] COM port set correctly in wasp_backend.py

---

## :scroll: License

This project was developed for UTM FAI Showcase 2026.

---

## :handshake: Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Open a Pull Request

---

## :busts_in_silhouette: Authors

| Role | Name |
|------|------|
| Backend / Integration | Uwais |
| IoT / Hardware | Paen |
| Computer Vision | Born |

---

*Last updated: June 2026*
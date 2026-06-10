"""
WASP - Warden Autonomous Safety Platform
MVP Backend for UTM FAI Showcase 2026
Team: Uwais (Integration/Agent), Paen (IoT), Born (CV)

Usage:
    1. Generate TTS audio: python generate_tts.py
    2. Update CONFIG section below
    3. Connect ESP32 to USB, XIAO to power (WiFi AP)
    4. python wasp_backend.py
    5. Open http://localhost:5000 in browser
    6. Connect laptop to "WASP-CAM-01" WiFi
"""

import cv2
import numpy as np
from flask import Flask, render_template_string, Response, jsonify
from ultralytics import YOLO
import sqlite3
import threading
import time
import requests
import os
import serial
import json
from datetime import datetime
from urllib.parse import quote

# ========================== CONFIG ==========================
# CHANGE THESE VALUES BEFORE RUNNING

# XIAO Camera Stream (AP Mode)
XIAO_IP = "192.168.4.1"           # Default AP IP for ESP32
XIAO_STREAM_URL = f"http://{XIAO_IP}:81/stream"

# ESP32 Serial Connection
ESP32_PORT = "COM3"               # CHANGE: Check Device Manager for your port
ESP32_BAUD = 115200

# YOLOv8 Model Path (Born trained model)
MODEL_PATH = "best.pt"            # CHANGE if your model file is named differently

# WhatsApp CallMeBot Settings
# Get API key from: https://www.callmebot.com/blog/free-api-whatsapp-messages/
SUPERVISOR_PHONE = "60123456789"  # CHANGE: Format 601xxxxxxxxx (no +, no dashes)
CALLMEBOT_KEY = "YOUR_APIKEY"     # CHANGE: Your CallMeBot API key

# Agent Settings
WARNING_COOLDOWN = 30             # Seconds before WhatsApp escalation
HEAT_THRESHOLD = 35.0             # Celsius - heat stress warning

# ========================== GLOBALS ==========================
app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()

sensor_data = {
    "temperature": 0.0,
    "humidity": 0.0,
    "motion": 0,
    "ir": 0,
    "sound": 0,
    "air_quality": 0,
    "last_update": "N/A"
}

cv_state = {
    "helmet": False,
    "harness": False,
    "person": False,
    "last_update": "N/A"
}

active_warnings = {}
model = None
tts_available = False
simulation_mode = False

# ========================== DATABASE ==========================
def init_db():
    """Initialize SQLite database for incident logging"""
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, alert_type TEXT, details TEXT, status TEXT DEFAULT 'ACTIVE')")
        conn.commit()
        conn.close()
        print("[DB] Database initialized")
    except Exception as e:
        print(f"[DB Error] {e}")

def log_alert(alert_type, details):
    """Log an alert to SQLite with timestamp"""
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("INSERT INTO alerts (timestamp, alert_type, details) VALUES (?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type, details))
        conn.commit()
        conn.close()
        print(f"[LOG] {alert_type}: {details}")
    except Exception as e:
        print(f"[LOG Error] {e}")

# ========================== TTS ==========================
def init_tts():
    """Initialize pygame mixer for audio playback"""
    global tts_available
    try:
        import pygame
        pygame.mixer.init()
        tts_available = True
        print("[TTS] Audio system ready")
    except Exception as e:
        print(f"[TTS] Audio unavailable: {e}")
        print("[TTS] Will print warnings to console instead")

def speak(text):
    """Play pre-generated warning MP3 or print to console"""
    if not tts_available:
        print(f"[TTS] {text}")
        return
    try:
        import pygame
        safe_name = text.replace(" ", "_").replace("!", "").replace(".", "").replace(":", "")[:50]
        path = f"warnings/{safe_name}.mp3"
        if os.path.exists(path):
            pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            print(f"[TTS] Playing: {safe_name}.mp3")
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        else:
            print(f"[TTS] Audio file not found: {path}")
            print(f"[TTS] {text}")
    except Exception as e:
        print(f"[TTS Error] {e}")
        print(f"[TTS] {text}")

# ========================== WHATSAPP ==========================
def whatsapp(msg):
    """Send WhatsApp message via CallMeBot API"""
    try:
        url = (
            f"https://api.callmebot.com/whatsapp.php?"
            f"phone={SUPERVISOR_PHONE}&"
            f"text={quote(msg)}&"
            f"apikey={CALLMEBOT_KEY}"
        )
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            print(f"[WhatsApp] Sent: {msg}")
        else:
            print(f"[WhatsApp] Failed: HTTP {r.status_code}")
    except Exception as e:
        print(f"[WhatsApp Error] {e}")
        print(f"[WhatsApp] Would have sent: {msg}")

# ========================== AGENTIC ENGINE ==========================
def agent_engine():
    """
    WASP Decision Engine - Tiered Response Protocol
    Tier 1: Verbal warning to worker (immediate)
    Tier 2: Supervisor WhatsApp alert (after 30 seconds non-compliance)
    Tier 3: Critical environmental alert (immediate for heat stress)
    """
    global active_warnings
    current_time = time.time()

    # PPE Violation Detection
    ppe_violations = []
    if cv_state["person"]:
        if not cv_state["helmet"]:
            ppe_violations.append("Helmet")
        if not cv_state["harness"]:
            ppe_violations.append("Harness")

    # Handle PPE violations
    if ppe_violations:
        v_key = "PPE_" + "_".join(ppe_violations)
        if v_key not in active_warnings:
            # TIER 1: First detection - verbal warning
            active_warnings[v_key] = current_time
            items = " dan ".join(ppe_violations)
            msg = f"Perhatian! {items} tidak dipakai. Sila pakai sekarang!"
            speak(msg)
            log_alert("PPE_VIOLATION", f"Missing: {', '.join(ppe_violations)}")
        # TIER 2: Escalation after cooldown period
        elapsed = current_time - active_warnings[v_key]
        if elapsed > WARNING_COOLDOWN and not active_warnings.get(v_key + "_esc"):
            active_warnings[v_key + "_esc"] = True
            msg = f"URGENT: Worker still missing {', '.join(ppe_violations)} after {WARNING_COOLDOWN} seconds!"
            whatsapp(msg)
            log_alert("ESCALATION", msg)
    else:
        # Clear PPE warnings when worker becomes compliant
        for k in list(active_warnings.keys()):
            if k.startswith("PPE_"):
                del active_warnings[k]

    # Heat Stress Detection (Critical - Immediate)
    if sensor_data["temperature"] > HEAT_THRESHOLD:
        v_key = "HEAT"
        if v_key not in active_warnings:
            active_warnings[v_key] = current_time
            msg = f"Perhatian! Suhu sangat tinggi: {sensor_data['temperature']:.1f}C. Sila berehat!"
            speak(msg)
            log_alert("HEAT_STRESS", msg)
            # Heat stress is critical - WhatsApp immediately
            whatsapp(f"HEAT STRESS ALERT: Zone temperature {sensor_data['temperature']:.1f}C")
    else:
        if "HEAT" in active_warnings:
            del active_warnings["HEAT"]

# ========================== CV THREAD ==========================
def cv_thread():
    """Continuously capture video and run YOLOv8 detection"""
    global latest_frame, cv_state

    print(f"[CV] Connecting to camera stream: {XIAO_STREAM_URL}")
    cap = cv2.VideoCapture(XIAO_STREAM_URL)

    # Fallback to laptop webcam if XIAO stream fails
    if not cap.isOpened():
        print("[CV] XIAO stream failed! Falling back to laptop webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CV] ERROR: No camera available!")
            return

    print("[CV] Camera connected successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        try:
            # Run YOLOv8 inference
            results = model(frame, conf=0.5, verbose=False)
            annotated = results[0].plot()

            # Parse detections (supports multiple naming conventions)
            helmet = harness = person = False
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = model.names[cls].lower()

                    if any(x in name for x in ["helmet", "hardhat", "hard_hat", "head"]):
                        helmet = True
                    elif any(x in name for x in ["harness", "safety_belt", "belt"]):
                        harness = True
                    elif any(x in name for x in ["person", "worker", "people", "man"]):
                        person = True

            cv_state = {
                "helmet": helmet,
                "harness": harness,
                "person": person,
                "last_update": datetime.now().strftime("%H:%M:%S")
            }

            # Update global frame for video feed
            with frame_lock:
                latest_frame = annotated.copy()

            # Run agent logic
            agent_engine()

        except Exception as e:
            print(f"[CV Error] {e}")

        time.sleep(0.3)

# ========================== SENSOR THREAD ==========================
def sensor_thread():
    """Read sensor data from ESP32 via serial USB"""
    global sensor_data, simulation_mode

    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        print(f"[Sensor] Connected to {ESP32_PORT} at {ESP32_BAUD} baud")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    data = json.loads(line)
                    sensor_data.update({
                        "temperature": data.get("temperature", 0),
                        "humidity": data.get("humidity", 0),
                        "motion": data.get("motion", 0),
                        "ir": data.get("ir", 0),
                        "sound": data.get("sound", 0),
                        "air_quality": data.get("air_quality", 0),
                        "last_update": datetime.now().strftime("%H:%M:%S")
                    })
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        print(f"[Sensor Error] {e}")
        print("[Sensor] Running in SIMULATION mode")
        simulation_mode = True

        # Simulation mode: generate fake data for testing
        while True:
            sensor_data.update({
                "temperature": 32.0 + (5 if (int(time.time()) % 20 > 10) else 0),
                "humidity": 65.0,
                "motion": 1 if (int(time.time()) % 5 > 2) else 0,
                "ir": 0,
                "sound": 0,
                "air_quality": 400,
                "last_update": datetime.now().strftime("%H:%M:%S")
            })
            time.sleep(2)

# ========================== DASHBOARD HTML ==========================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WASP - Safety Monitor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: "Segoe UI", Arial, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); padding: 20px; text-align: center; font-size: 28px; font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.3); letter-spacing: 2px; }
        .header span { color: #fbbf24; }
        .alert-banner { background: #dc2626; padding: 15px; text-align: center; font-size: 20px; font-weight: bold; display: none; animation: pulse 1s infinite; border-bottom: 3px solid #fbbf24; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 20px; max-width: 1400px; margin: 0 auto; }
        .panel { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        .panel h3 { margin-top: 0; color: #38bdf8; font-size: 18px; margin-bottom: 15px; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        .video-container img { width: 100%; border-radius: 8px; border: 2px solid #334155; }
        .sensor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .sensor-box { background: #0f172a; padding: 15px; text-align: center; border-radius: 8px; border: 1px solid #334155; }
        .sensor-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .sensor-value { font-size: 28px; font-weight: bold; color: #38bdf8; margin-top: 5px; }
        .status-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #334155; }
        .status-label { color: #94a3b8; }
        .status-safe { color: #22c55e; font-weight: bold; }
        .status-warn { color: #f59e0b; font-weight: bold; }
        .status-danger { color: #ef4444; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #334155; }
        th { color: #38bdf8; font-weight: 600; background: #0f172a; }
        td { color: #cbd5e1; }
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; text-transform: uppercase; }
        .badge-ppe { background: #f59e0b; color: #0f172a; }
        .badge-heat { background: #dc2626; color: white; }
        .badge-esc { background: #7c3aed; color: white; }
        .footer { text-align: center; padding: 20px; color: #64748b; font-size: 12px; border-top: 1px solid #334155; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="header">🛡️ <span>WASP</span> — Warden Autonomous Safety Platform</div>
    <div id="alert-banner" class="alert-banner">⚠️ VIOLATION DETECTED — SPEAKING WARNING ⚠️</div>
    <div class="grid">
        <div class="panel video-container">
            <h3>📹 Live Feed — Zone A</h3>
            <img src="/video_feed" alt="Camera Feed" onerror="this.src='data:image/svg+xml,&lt;svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22&gt;&lt;rect fill=%22%23334155%22 width=%22400%22 height=%22300%22/&gt;&lt;text fill=%22%2394a3b8%22 x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22&gt;Camera Offline&lt;/text&gt;&lt;/svg&gt;'">
        </div>
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="panel">
                <h3>🌡️ Environmental Sensors</h3>
                <div class="sensor-grid">
                    <div class="sensor-box"><div class="sensor-label">Temperature</div><div class="sensor-value" id="temp">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Humidity</div><div class="sensor-value" id="humid">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Motion</div><div class="sensor-value" id="motion">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Status</div><div class="sensor-value" id="status">--</div></div>
                </div>
            </div>
            <div class="panel">
                <h3>👁️ Computer Vision Detection</h3>
                <div class="status-row"><span class="status-label">Helmet Detected</span><span id="helmet" class="status-safe">--</span></div>
                <div class="status-row"><span class="status-label">Harness Detected</span><span id="harness" class="status-safe">--</span></div>
                <div class="status-row"><span class="status-label">Person Detected</span><span id="person" class="status-safe">--</span></div>
                <div class="status-row" style="border-bottom: none;"><span class="status-label">Last Update</span><span id="cv-update" style="color: #64748b; font-size: 12px;">--</span></div>
            </div>
        </div>
    </div>
    <div style="max-width: 1400px; margin: 0 auto; padding: 0 20px 20px;">
        <div class="panel">
            <h3>🚨 Recent Alerts</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr><th>Time</th><th>Type</th><th>Details</th><th>Status</th></tr></thead>
                    <tbody id="alerts-body"></tbody>
                </table>
            </div>
        </div>
    </div>
    <div class="footer">WASP MVP — UTM FAI Showcase 2026 | Running in <span id="mode">Live</span> Mode</div>
    <script>
        async function updateData() {
            try {
                const s = await fetch('/api/sensors').then(r => r.json());
                document.getElementById('temp').textContent = (s.temperature || 0).toFixed(1) + '°C';
                document.getElementById('humid').textContent = (s.humidity || 0).toFixed(1) + '%';
                document.getElementById('motion').textContent = s.motion ? 'DETECTED' : 'CLEAR';
                document.getElementById('motion').style.color = s.motion ? '#f59e0b' : '#22c55e';
                const c = await fetch('/api/cv').then(r => r.json());
                const helmetEl = document.getElementById('helmet');
                const harnessEl = document.getElementById('harness');
                const personEl = document.getElementById('person');
                helmetEl.textContent = c.helmet ? '✅ YES' : '❌ NO';
                helmetEl.className = c.helmet ? 'status-safe' : 'status-danger';
                harnessEl.textContent = c.harness ? '✅ YES' : '❌ NO';
                harnessEl.className = c.harness ? 'status-safe' : 'status-danger';
                personEl.textContent = c.person ? '✅ YES' : '❌ NO';
                personEl.className = c.person ? 'status-safe' : 'status-warn';
                document.getElementById('cv-update').textContent = c.last_update || 'N/A';
                const hasViolation = c.person && (!c.helmet || !c.harness);
                const heatStress = (s.temperature || 0) > 35.0;
                const statusEl = document.getElementById('status');
                if (heatStress) { statusEl.textContent = '🔥 HEAT'; statusEl.style.color = '#dc2626'; }
                else if (hasViolation) { statusEl.textContent = '⚠️ WARN'; statusEl.style.color = '#f59e0b'; }
                else { statusEl.textContent = '✅ SAFE'; statusEl.style.color = '#22c55e'; }
                document.getElementById('alert-banner').style.display = (hasViolation || heatStress) ? 'block' : 'none';
                const a = await fetch('/api/alerts').then(r => r.json());
                const tbody = document.getElementById('alerts-body');
                tbody.innerHTML = a.slice(0, 10).map(row => {
                    let badgeClass = 'badge-ppe';
                    if (row.alert_type.includes('HEAT')) badgeClass = 'badge-heat';
                    if (row.alert_type.includes('ESCALATION')) badgeClass = 'badge-esc';
                    return `<tr><td>${row.timestamp}</td><td><span class="badge ${badgeClass}">${row.alert_type}</span></td><td>${row.details}</td><td>${row.status}</td></tr>`;
                }).join('');
            } catch(e) { console.error('Update error:', e); }
        }
        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""

# ========================== FLASK ROUTES ==========================
@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    """Stream annotated video feed from CV thread"""
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors')
def api_sensors():
    """Return latest sensor data as JSON"""
    return jsonify(sensor_data)

@app.route('/api/cv')
def api_cv():
    """Return latest CV detection state as JSON"""
    return jsonify(cv_state)

@app.route('/api/alerts')
def api_alerts():
    """Return recent alerts from database"""
    try:
        conn = sqlite3.connect('wasp.db', check_same_thread=False)
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20")
        rows = c.fetchall()
        conn.close()
        return jsonify([{
            "id": r[0],
            "timestamp": r[1],
            "alert_type": r[2],
            "details": r[3],
            "status": r[4]
        } for r in rows])
    except Exception as e:
        print(f"[API Error] {e}")
        return jsonify([])

# ========================== MAIN ==========================
if __name__ == '__main__':
    print("=" * 60)
    print(" WASP - Warden Autonomous Safety Platform")
    print(" MVP Build - UTM FAI Showcase 2026")
    print("=" * 60)

    os.makedirs("warnings", exist_ok=True)

    init_db()
    init_tts()

    print(f"[INIT] Loading YOLOv8 model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"[INIT] Model loaded. Classes: {list(model.names.values())}")
    except Exception as e:
        print(f"[INIT] FATAL: Cannot load model: {e}")
        print("[INIT] Make sure 'best.pt' is in the same folder as this script")
        exit(1)

    print("[INIT] Starting CV thread...")
    threading.Thread(target=cv_thread, daemon=True).start()

    print("[INIT] Starting Sensor thread...")
    threading.Thread(target=sensor_thread, daemon=True).start()

    print("[INIT] Starting web server...")
    print("[INIT] Dashboard: http://localhost:5000")
    print("[INIT] Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

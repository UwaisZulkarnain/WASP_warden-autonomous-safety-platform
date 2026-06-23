"""
WASP - Warden Autonomous Safety Platform
MVP Backend for UTM FAI Showcase 2026
Team: Uwais (Backend/Integration), Paen (IoT), Born (CV)

Advanced version with per-person PPE spatial matching.
Each person is tracked individually. PPE items are matched to specific
people using bounding box containment (is_inside logic).

Usage:
    1. Generate TTS audio: python tts/generate_tts.py
    2. Update CONFIG section below (COM port, phone, API key)
    3. Connect ESP32 to USB, webcam to laptop
    4. python wasp_backend.py
    5. Open http://localhost:5000 in browser
"""
 #TETSTSTSTSTTSTSTSTSTTSTS
import cv2
import numpy as np
from flask import Flask, render_template_string, Response, jsonify, request
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
import torch
import paho.mqtt.client as mqtt
import json
CUDA_AVAILABLE = torch.cuda.is_available()

# ========================== CONFIG ==========================
# CHANGE THESE VALUES BEFORE RUNNING
# Camera source: 0 = laptop webcam, or XIAO stream URL
USE_XIAO = False
XIAO_IP = "192.168.4.1"
XIAO_STREAM_URL = f"http://{XIAO_IP}:81/stream"

# ESP32 Serial Connection
ESP32_PORT = "COM3"
ESP32_BAUD = 115200

# YOLOv8 Model Path (Born trained model)
MODEL_PATH = "newbest.pt"
SIMULATE_CV = False   # True = fake CV data, False = real camera YOLO

# WhatsApp CallMeBot Settings
# Get API key from: https://www.callmebot.com/blog/free-api-whatsapp-messages/
SUPERVISOR_PHONE = "60123456789"
CALLMEBOT_KEY = "YOUR_APIKEY"

# Agent Settings
WARNING_COOLDOWN = 30
HEAT_THRESHOLD = 35.0
CONFIDENCE_THRESHOLD = 0.30

# CV output mode
# "full"  = check helmet + vest
# "head"  = check helmet only
# "torso" = check vest only
# "auto"  = system estimates visible segment automatically
CURRENT_SEGMENT = "full"
SHOW_SIMPLE_VIEW = True
VIOLATION_CONFIRM_FRAMES = 5

BROKER_HOST = "localhost"   #recommended localhost if same device
BROKER_PORT = 1883          #depends on running device port
TOPIC       = "sensors/#"
USERNAME    = ""            # if available
PASSWORD    = ""            # if available

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

# Per-person CV state with spatial matching
cv_state = {
    "person_count": 0,
    "persons": [],
    "any_violation": False,
    "main_compliance": True,
    "inspection_mode": CURRENT_SEGMENT,
    "global_helmet": True,
    "global_vest": True,
    "global_goggles": True,
    "global_gloves": True,
    "global_boots": True,
    "last_update": "N/A"
}

active_warnings = {}
segment_mode = CURRENT_SEGMENT
violation_streak = 0
safe_streak = 0
model = None
tts_available = False
simulation_mode = False

# ========================== MQTT Reciever ========================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
    else:
        print(f"Connection failed, reason code: {reason_code}")


def on_message(client, userdata, msg):
    global sensor_data
    topic   = msg.topic
    payload = msg.payload.decode("utf-8")

    try:
        data = json.loads(payload)
        temp     = data['temperature']
        humidity = data['humidity']
        motion   = data['motion']
        obstacle = data['obstacle']
        mic_level = data['mic_level']
        mq2_raw  = data['mq2_raw']

        sensor_data.update({
            "temperature": temp,
            "humidity":    humidity,
            "motion":      1 if motion else 0,
            "ir":          1 if obstacle else 0,
            "sound":       mic_level,
            "air_quality": mq2_raw,
            "last_update": datetime.now().strftime("%H:%M:%S")
        })
        print(f"{motion}    {obstacle}      {humidity}      {temp}      {mic_level}     {mq2_raw}")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[MQTT] Bad payload: {e}")

    



def on_disconnect(client, userdata, flags, reason_code, properties):
    print(f"Disconnected ({reason_code})")

# ==============================================================

# ========================== DATABASE ==========================
def init_db():
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
    if not tts_available:
        print(f"[TTS] {text}")
        return
    try:
        import pygame
        safe_name = text.replace(" ", "_").replace("!", "").replace(".", "").replace(":", "")
        path = f"warnings/{safe_name}.mp3"
        if os.path.exists(path):
            def play_async():
                try:
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    print(f"[TTS] Playing: {safe_name}.mp3")
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                except Exception as e:
                    print(f"[TTS Error] {e}")
            threading.Thread(target=play_async, daemon=True).start()
        else:
            print(f"[TTS] Audio file not found: {path}")
            print(f"[TTS] {text}")
    except Exception as e:
        print(f"[TTS Error] {e}")
        print(f"[TTS] {text}")

# ========================== WHATSAPP ==========================
def whatsapp(msg):
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

# ========================== SPATIAL MATCHING (Born) ==========================
def normalize_detected_class(name):
    """
    Normalize class names from different trained models.
    This keeps the backend stable even if the model class names are slightly different.
    """
    raw = str(name).lower().strip()
    compact = raw.replace(" ", "").replace("_", "").replace("-", "")

    if compact in ["person", "worker", "human"]:
        return "person"
    if compact in ["helmet", "hardhat", "safetyhelmet"]:
        return "helmet"
    if compact in ["vest", "safetyvest", "reflectivevest", "jacket"]:
        return "vest"
    if compact in ["goggles", "goggle", "safetygoggles", "glasses", "safetyglasses"]:
        return "goggles"
    if compact in ["gloves", "glove", "safetygloves"]:
        return "gloves"
    if compact in ["boots", "boot", "safetyboots", "shoes", "safetyshoes"]:
        return "boots"

    return raw


def is_inside(inner_box, outer_box):
    """
    Check if PPE box is inside person box.
    box format: [x1, y1, x2, y2]
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    center_x = (ix1 + ix2) / 2
    center_y = (iy1 + iy2) / 2
    return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2


def expand_box(box, margin=25):
    """
    Expand a bounding box slightly to make PPE-person matching more forgiving.
    box format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - margin),
        max(0, y1 - margin),
        x2 + margin,
        y2 + margin
    ]


def get_body_zones(person_box):
    """
    Split a person bounding box into simple body zones.
    This is not true segmentation, but it improves PPE matching.

    box format: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = person_box
    height = max(1, y2 - y1)

    head_zone = [
        x1,
        y1,
        x2,
        y1 + int(height * 0.35)
    ]

    torso_zone = [
        x1,
        y1 + int(height * 0.25),
        x2,
        y1 + int(height * 0.78)
    ]

    foot_zone = [
        x1,
        y1 + int(height * 0.65),
        x2,
        y2
    ]

    return {
        "head": head_zone,
        "torso": torso_zone,
        "foot": foot_zone
    }


def match_ppe_to_zone(ppe_box, zone_box):
    """
    Check whether the center point of a PPE box is inside a body zone.
    """
    px1, py1, px2, py2 = ppe_box
    zx1, zy1, zx2, zy2 = zone_box

    center_x = (px1 + px2) / 2
    center_y = (py1 + py2) / 2

    return zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2


def get_required_ppe(effective_segment):
    """
    Decide which PPE should be checked for the current visible segment.
    This prevents false alerts for PPE that is outside the camera view.
    """
    if effective_segment == "head":
        return ["helmet"]
    if effective_segment == "torso":
        return ["vest"]
    return ["helmet", "vest"]


def get_effective_segment(person_box, frame_shape):
    """
    Decide which segment is currently being inspected.
    If CURRENT_SEGMENT is not auto, use the selected mode directly.
    """
    global segment_mode

    mode = str(segment_mode).lower().strip()
    if mode in ["head", "torso", "full"]:
        return mode

    # Auto mode heuristic. This is intentionally simple for showcase stability.
    frame_h = frame_shape[0]
    x1, y1, x2, y2 = person_box
    person_h = max(1, y2 - y1)
    visible_ratio = person_h / max(1, frame_h)

    # Close-up upper face/head view usually gives a smaller visible body region.
    if visible_ratio < 0.50:
        return "head"

    # Most webcam views show head + torso but not feet, so torso/full PPE is reasonable.
    if visible_ratio >= 0.65:
        return "full"

    return "torso"


def draw_clean_annotations(frame, person_outputs):
    """
    Draw a simple CV output on the monitor.
    Only person, helmet, and vest are shown to avoid a complex noisy display.
    """
    for person in person_outputs:
        x1, y1, x2, y2 = person["bbox"]
        status = person.get("status", "SAFE")
        mode = person.get("effective_segment", "full").upper()

        if status == "VIOLATION":
            color = (0, 0, 255)
        elif status == "CHECKING":
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{person['person_id']} | {mode} | {status}"
        cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw only required PPE boxes, not every detected class.
        for item_name, item_box in person.get("display_boxes", {}).items():
            ix1, iy1, ix2, iy2 = item_box
            ppe_color = (255, 180, 0) if item_name == "helmet" else (0, 180, 255)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), ppe_color, 2)
            cv2.putText(frame, item_name.upper(), (ix1, max(25, iy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ppe_color, 2)

    return frame

# ========================== DECISION ENGINE ==========================
def agent_engine():
    """
    Tiered response protocol based on per-person violation analysis.
    Tier 1: Verbal warning to worker (immediate)
    Tier 2: Supervisor WhatsApp alert (after 30 seconds non-compliance)
    Tier 3: Critical environmental alert (immediate for heat stress)
    """
    global active_warnings
    current_time = time.time()

    # Build list of all missing PPE types across all persons
    missing_types = set()
    violation_details = []
    for person in cv_state["persons"]:
        if person["status"] == "VIOLATION":
            for v in person["violations"]:
                missing_types.add(v)
            violation_details.append(f"{person['person_id']}: {', '.join(person['violations'])}")

    # PPE Violation Detection
    if missing_types:
        v_key = "PPE_" + "_".join(sorted(missing_types))
        if v_key not in active_warnings:
            active_warnings[v_key] = current_time
            # Build warning message from missing types
            items = []
            if "Missing helmet" in missing_types:
                items.append("Helmet")
            if "Missing vest" in missing_types:
                items.append("Vest")
            if not items:
                items = ["PPE"]
            items_str = " dan ".join(items)
            msg = f"Perhatian! {items_str} tidak dikesan. Sila semak PPE sekarang!"
            speak(msg)
            log_alert("PPE_VIOLATION", "; ".join(violation_details))
        # Tier 2: Escalation after cooldown
        elapsed = current_time - active_warnings[v_key]
        if elapsed > WARNING_COOLDOWN and not active_warnings.get(v_key + "_esc"):
            active_warnings[v_key + "_esc"] = True
            msg = f"URGENT: {cv_state['person_count']} worker(s) still non-compliant after {WARNING_COOLDOWN}s: {', '.join(missing_types)}"
            whatsapp(msg)
            log_alert("ESCALATION", msg)
    else:
        for k in list(active_warnings.keys()):
            if k.startswith("PPE_"):
                del active_warnings[k]

    # Heat Stress Detection (Critical - Immediate)
    if sensor_data["temperature"] > HEAT_THRESHOLD:
        v_key = "HEAT"
        if v_key not in active_warnings:
            active_warnings[v_key] = current_time
            msg = "Perhatian! Suhu sangat tinggi. Sila berehat!"
            speak(msg)
            log_alert("HEAT_STRESS", msg)
            whatsapp(f"HEAT STRESS ALERT: Zone temperature {sensor_data['temperature']:.1f}C")
    else:
        if "HEAT" in active_warnings:
            del active_warnings["HEAT"]

# ========================== CV THREAD (Born logic) ==========================
def cv_thread():
    global latest_frame, cv_state, violation_streak, safe_streak

    # ==========================
    # CV SIMULATION MODE
    # ==========================
    if SIMULATE_CV:
        print("[CV] Running in SIMULATION mode")

        scenario = 0

        while True:
            scenario = (scenario + 1) % 3

            # Scenario 1: No person
            if scenario == 0:
                cv_state = {
                    "person_count": 0,
                    "persons": [],
                    "any_violation": False,
                    "main_compliance": True,
                    "inspection_mode": segment_mode,
                    "global_helmet": True,
                    "global_vest": True,
                    "global_goggles": True,
                    "global_gloves": True,
                    "global_boots": True,
                    "last_update": datetime.now().strftime("%H:%M:%S")
                }

            # Scenario 2: Safe worker
            elif scenario == 1:
                cv_state = {
                    "person_count": 1,
                    "persons": [
                        {
                            "person_id": "Worker 1",
                            "bbox": [120, 70, 420, 480],
                            "confidence": 0.96,
                            "effective_segment": "full",
                            "required_ppe": ["helmet", "vest"],
                            "ppe": {
                                "helmet": True,
                                "vest": True,
                                "goggles": False,
                                "gloves": False,
                                "boots": False
                            },
                            "ppe_confidence": {
                                "helmet": 0.95,
                                "vest": 0.92,
                                "goggles": 0.0,
                                "gloves": 0.0,
                                "boots": 0.0
                            },
                            "violations": [],
                            "not_checked": ["goggles", "gloves", "boots"],
                            "status": "SAFE"
                        }
                    ],
                    "any_violation": False,
                    "main_compliance": True,
                    "inspection_mode": segment_mode,
                    "global_helmet": True,
                    "global_vest": True,
                    "global_goggles": True,
                    "global_gloves": True,
                    "global_boots": True,
                    "last_update": datetime.now().strftime("%H:%M:%S")
                }

            # Scenario 3: PPE violation
            else:
                cv_state = {
                    "person_count": 1,
                    "persons": [
                        {
                            "person_id": "Worker 1",
                            "bbox": [120, 70, 420, 480],
                            "confidence": 0.94,
                            "effective_segment": "full",
                            "required_ppe": ["helmet", "vest"],
                            "ppe": {
                                "helmet": False,
                                "vest": True,
                                "goggles": False,
                                "gloves": False,
                                "boots": False
                            },
                            "ppe_confidence": {
                                "helmet": 0.0,
                                "vest": 0.91,
                                "goggles": 0.0,
                                "gloves": 0.0,
                                "boots": 0.0
                            },
                            "violations": ["Missing helmet"],
                            "not_checked": ["goggles", "gloves", "boots"],
                            "status": "VIOLATION"
                        }
                    ],
                    "any_violation": True,
                    "main_compliance": False,
                    "inspection_mode": segment_mode,
                    "global_helmet": False,
                    "global_vest": True,
                    "global_goggles": True,
                    "global_gloves": True,
                    "global_boots": True,
                    "last_update": datetime.now().strftime("%H:%M:%S")
                }

            print("[CV SIMULATION JSON]")
            print(json.dumps(cv_state, indent=4))

            agent_engine()
            time.sleep(5)

    # ==========================
    # REAL YOLO CAMERA MODE
    # ==========================
    if USE_XIAO:
        print(f"[CV] Connecting to XIAO stream: {XIAO_STREAM_URL}")
        cap = cv2.VideoCapture(XIAO_STREAM_URL)
    else:
        print("[CV] Using laptop webcam")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        if USE_XIAO:
            print("[CV] XIAO stream failed! Falling back to laptop webcam...")
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[CV] ERROR: No camera available!")
            return

    print("[CV] Camera connected successfully")

    fps_counter = 0
    fps_start = time.time()
    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.5)
            continue

        frame_counter += 1

        if not CUDA_AVAILABLE and frame_counter % 2 != 0:
            continue

        try:
            results = model(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
                imgsz=640
            )

            annotated = frame.copy() if SHOW_SIMPLE_VIEW else results[0].plot()

            persons = []
            ppe_items = []

            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = normalize_detected_class(results[0].names[cls])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [round(x1), round(y1), round(x2), round(y2)]

                detection = {
                    "class_name": name,
                    "confidence": round(conf, 2),
                    "bbox": bbox
                }

                if name == "person":
                    persons.append(detection)
                elif name in ["helmet", "vest", "goggles", "gloves", "boots"]:
                    ppe_items.append(detection)

            person_outputs = []

            for idx, person in enumerate(persons, start=1):
                p_box = person["bbox"]
                expanded_person_box = expand_box(p_box, margin=30)
                body_zones = get_body_zones(expanded_person_box)
                effective_segment = get_effective_segment(p_box, frame.shape)
                required_ppe = get_required_ppe(effective_segment)

                ppe_status = {
                    "helmet": False,
                    "vest": False,
                    "goggles": False,
                    "gloves": False,
                    "boots": False,
                }

                ppe_confidence = {
                    "helmet": 0.0,
                    "vest": 0.0,
                    "goggles": 0.0,
                    "gloves": 0.0,
                    "boots": 0.0,
                }

                display_boxes = {}

                for item in ppe_items:
                    item_name = item["class_name"]
                    item_box = item["bbox"]

                    if item_name not in ppe_status:
                        continue

                    matched = False

                    # Helmet and goggles should be around the head area
                    if item_name in ["helmet", "goggles"]:
                        matched = match_ppe_to_zone(item_box, body_zones["head"])

                    # Vest should be around the torso area
                    elif item_name == "vest":
                        matched = match_ppe_to_zone(item_box, body_zones["torso"])

                    # Boots are optional because full-body camera angle is difficult
                    elif item_name == "boots":
                        matched = match_ppe_to_zone(item_box, body_zones["foot"])

                    # Gloves are kept for dashboard consistency only
                    elif item_name == "gloves":
                        matched = is_inside(item_box, expanded_person_box)

                    if matched:
                        ppe_status[item_name] = True
                        ppe_confidence[item_name] = max(
                            ppe_confidence[item_name],
                            item["confidence"]
                        )
                        if item_name in required_ppe:
                            display_boxes[item_name] = item_box

                violations = []

                # Segment-aware violation rule.
                # This avoids false alerts for PPE that is outside the current visible segment.
                if "helmet" in required_ppe and not ppe_status["helmet"]:
                    violations.append("Missing helmet")

                if "vest" in required_ppe and not ppe_status["vest"]:
                    violations.append("Missing vest")

                not_checked = [
                    item for item in ["helmet", "vest", "goggles", "gloves", "boots"]
                    if item not in required_ppe
                ]

                status = "SAFE" if len(violations) == 0 else "VIOLATION"

                person_outputs.append({
                    "person_id": f"Worker {idx}",
                    "bbox": p_box,
                    "confidence": person["confidence"],
                    "effective_segment": effective_segment,
                    "required_ppe": required_ppe,
                    "ppe": ppe_status,
                    "ppe_confidence": ppe_confidence,
                    "display_boxes": display_boxes,
                    "violations": violations,
                    "not_checked": not_checked,
                    "status": status
                })

            raw_violation = any(p["status"] == "VIOLATION" for p in person_outputs)

            # Small confirmation buffer to avoid flickering alerts from one bad frame.
            if raw_violation:
                violation_streak += 1
                safe_streak = 0
            else:
                safe_streak += 1
                violation_streak = 0

            confirmed_violation = raw_violation and violation_streak >= VIOLATION_CONFIRM_FRAMES

            if raw_violation and not confirmed_violation:
                for p in person_outputs:
                    if p["status"] == "VIOLATION":
                        p["status"] = "CHECKING"

            any_violation = confirmed_violation
            main_compliance = not confirmed_violation

            g_helmet = all(p["ppe"]["helmet"] for p in person_outputs) if person_outputs else True
            g_vest = all(p["ppe"]["vest"] for p in person_outputs) if person_outputs else True
            g_goggles = all(p["ppe"]["goggles"] for p in person_outputs) if person_outputs else True
            g_gloves = all(p["ppe"]["gloves"] for p in person_outputs) if person_outputs else True
            g_boots = all(p["ppe"]["boots"] for p in person_outputs) if person_outputs else True

            if SHOW_SIMPLE_VIEW:
                annotated = draw_clean_annotations(annotated, person_outputs)

            cv_state = {
                "person_count": len(person_outputs),
                "persons": person_outputs,
                "any_violation": any_violation,
                "main_compliance": main_compliance,
                "inspection_mode": segment_mode,
                "global_helmet": g_helmet,
                "global_vest": g_vest,
                "global_goggles": g_goggles,
                "global_gloves": g_gloves,
                "global_boots": g_boots,
                "last_update": datetime.now().strftime("%H:%M:%S")
            }

            with frame_lock:
                latest_frame = annotated.copy()

            agent_engine()

            fps_counter += 1

            if time.time() - fps_start >= 5.0:
                print(f"[CV] FPS: {fps_counter / 5.0:.1f}")
                print("[CV JSON]")
                print(json.dumps(cv_state, indent=4))
                fps_counter = 0
                fps_start = time.time()

        except Exception as e:
            print(f"[CV Error] {e}")

        time.sleep(0.01)

# ========================== SENSOR THREAD ==========================
def sensor_thread():
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
        .alert-banner { background: #dc2626; padding: 14px; text-align: center; font-size: 18px; font-weight: bold; display: none; animation: pulse 1s infinite; border-bottom: 3px solid #fbbf24; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.75; } }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 20px; max-width: 1400px; margin: 0 auto; }
        .panel { background: #1e293b; padding: 18px; border-radius: 12px; border: 1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
        .panel h3 { margin-top: 0; color: #38bdf8; font-size: 18px; margin-bottom: 14px; border-bottom: 2px solid #334155; padding-bottom: 10px; }
        .video-container img { width: 100%; border-radius: 8px; border: 2px solid #334155; }
        .status-card { background: #0f172a; padding: 15px; border-radius: 10px; border: 1px solid #334155; margin-bottom: 12px; }
        .status-label { font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
        .big-status { font-size: 30px; font-weight: bold; margin-top: 5px; }
        .safe { color: #22c55e; }
        .danger { color: #ef4444; }
        .checking { color: #f59e0b; }
        .muted { color: #94a3b8; }
        .mode-buttons { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
        .mode-buttons button { background: #334155; color: #e2e8f0; border: 1px solid #475569; padding: 9px; border-radius: 8px; cursor: pointer; font-weight: bold; }
        .mode-buttons button.active { background: #38bdf8; color: #0f172a; }
        .sensor-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .sensor-box { background: #0f172a; padding: 12px; text-align: center; border-radius: 8px; border: 1px solid #334155; }
        .sensor-value { font-size: 22px; font-weight: bold; color: #38bdf8; margin-top: 5px; }
        .person-card { background: #0f172a; padding: 15px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #334155; }
        .person-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .person-id { font-weight: bold; color: #38bdf8; }
        .person-status { font-size: 12px; font-weight: bold; padding: 4px 10px; border-radius: 12px; }
        .person-status.safe { background: #22c55e; color: #0f172a; }
        .person-status.violation { background: #ef4444; color: white; }
        .person-status.checking { background: #f59e0b; color: #0f172a; }
        .ppe-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; }
        .ppe-item { text-align: center; padding: 12px; border-radius: 8px; font-size: 13px; font-weight: bold; }
        .ppe-ok { background: #22c55e33; color: #22c55e; border: 1px solid #22c55e; }
        .ppe-missing { background: #ef444433; color: #ef4444; border: 1px solid #ef4444; }
        .ppe-ignore { background: #334155; color: #94a3b8; border: 1px solid #475569; }
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
    <div class="header">WASP <span>Warden Autonomous Safety Platform</span></div>
    <div id="alert-banner" class="alert-banner">PPE VIOLATION CONFIRMED</div>

    <div class="grid">
        <div class="panel video-container">
            <h3>Live Feed - Clean CV Output</h3>
            <img src="/video_feed" alt="Camera Feed" onerror="this.src='data:image/svg+xml,&lt;svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22&gt;&lt;rect fill=%22%23334155%22 width=%22400%22 height=%22300%22/&gt;&lt;text fill=%22%2394a3b8%22 x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22&gt;Camera Offline&lt;/text&gt;&lt;/svg&gt;'">
        </div>

        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="panel">
                <h3>Safety Status</h3>
                <div class="status-card">
                    <div class="status-label">Current Status</div>
                    <div id="status" class="big-status muted">--</div>
                </div>
                <div class="status-card">
                    <div class="status-label">Inspection Mode</div>
                    <div id="mode-label" style="font-size:22px; font-weight:bold; margin-top:5px; color:#38bdf8;">--</div>
                    <div class="mode-buttons">
                        <button onclick="setMode('auto')" id="btn-auto">AUTO</button>
                        <button onclick="setMode('head')" id="btn-head">HEAD</button>
                        <button onclick="setMode('torso')" id="btn-torso">TORSO</button>
                        <button onclick="setMode('full')" id="btn-full">FULL</button>
                    </div>
                    <div style="font-size:12px; color:#94a3b8; margin-top:8px;">HEAD checks helmet only. TORSO checks vest only. FULL checks helmet and vest.</div>
                </div>
                <div class="status-card">
                    <div class="status-label">Workers Detected</div>
                    <div id="person-count" style="font-size:26px; font-weight:bold; margin-top:5px; color:#38bdf8;">0</div>
                </div>
            </div>

            <div class="panel">
                <h3>Environmental Sensors</h3>
                <div class="sensor-grid">
                    <div class="sensor-box"><div class="status-label">Temperature</div><div class="sensor-value" id="temp">--</div></div>
                    <div class="sensor-box"><div class="status-label">Humidity</div><div class="sensor-value" id="humid">--</div></div>
                    <div class="sensor-box"><div class="status-label">Motion</div><div class="sensor-value" id="motion">--</div></div>
                    <div class="sensor-box"><div class="status-label">Main PPE</div><div class="sensor-value" id="main-ppe">--</div></div>
                </div>
            </div>
        </div>
    </div>

    <div style="max-width: 1400px; margin: 0 auto; padding: 0 20px 20px;">
        <div class="panel">
            <h3>Per-Person PPE Check</h3>
            <div id="persons-container"></div>
        </div>
    </div>

    <div style="max-width: 1400px; margin: 0 auto; padding: 0 20px 20px;">
        <div class="panel">
            <h3>Recent Alerts</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr><th>Time</th><th>Type</th><th>Details</th><th>Status</th></tr></thead>
                    <tbody id="alerts-body"></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="footer">WASP MVP - UTM FAI Showcase 2026 | Clean CV Mode</div>

    <script>
        async function setMode(mode) {
            await fetch('/api/mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            });
            updateData();
        }

        function updateModeButtons(mode) {
            ['auto', 'head', 'torso', 'full'].forEach(m => {
                const btn = document.getElementById('btn-' + m);
                if (btn) btn.className = (m === mode) ? 'active' : '';
            });
        }

        function ppeBox(label, value, checked, conf) {
            if (!checked) {
                return `<div class="ppe-item ppe-ignore">${label}<br><span style="font-size:11px;">NOT CHECKED</span></div>`;
            }
            const cls = value ? 'ppe-ok' : 'ppe-missing';
            const text = value ? 'OK' : 'MISSING';
            const score = conf ? conf.toFixed(2) : '0.00';
            return `<div class="ppe-item ${cls}">${label}: ${text}<br><span style="font-size:11px;">${score}</span></div>`;
        }

        async function updateData() {
            try {
                const s = await fetch('/api/sensors').then(r => r.json());
                const c = await fetch('/api/cv').then(r => r.json());
                const m = await fetch('/api/mode').then(r => r.json());

                document.getElementById('temp').textContent = (s.temperature || 0).toFixed(1) + ' C';
                document.getElementById('humid').textContent = (s.humidity || 0).toFixed(1) + '%';
                document.getElementById('motion').textContent = s.motion ? 'YES' : 'NO';
                document.getElementById('person-count').textContent = c.person_count || 0;

                const mode = m.mode || c.inspection_mode || 'full';
                document.getElementById('mode-label').textContent = mode.toUpperCase();
                updateModeButtons(mode);

                const hasViolation = c.any_violation;
                const heatStress = (s.temperature || 0) > 35.0;
                const statusEl = document.getElementById('status');
                const mainPpe = document.getElementById('main-ppe');

                if (heatStress) {
                    statusEl.textContent = 'HEAT';
                    statusEl.className = 'big-status danger';
                } else if (hasViolation) {
                    statusEl.textContent = 'VIOLATION';
                    statusEl.className = 'big-status danger';
                } else if ((c.persons || []).some(p => p.status === 'CHECKING')) {
                    statusEl.textContent = 'CHECKING';
                    statusEl.className = 'big-status checking';
                } else {
                    statusEl.textContent = 'SAFE';
                    statusEl.className = 'big-status safe';
                }

                mainPpe.textContent = c.main_compliance ? 'OK' : 'MISS';
                mainPpe.style.color = c.main_compliance ? '#22c55e' : '#ef4444';
                document.getElementById('alert-banner').style.display = (hasViolation || heatStress) ? 'block' : 'none';

                const container = document.getElementById('persons-container');
                if (c.persons && c.persons.length > 0) {
                    container.innerHTML = c.persons.map(p => {
                        const ppe = p.ppe || {};
                        const conf = p.ppe_confidence || {};
                        const required = p.required_ppe || [];
                        const violations = p.violations && p.violations.length > 0 ? p.violations.join(', ') : 'None';
                        const statusClass = p.status === 'VIOLATION' ? 'violation' : (p.status === 'CHECKING' ? 'checking' : 'safe');

                        return `<div class="person-card">
                            <div class="person-header">
                                <span class="person-id">${p.person_id} | ${String(p.effective_segment || mode).toUpperCase()} MODE</span>
                                <span class="person-status ${statusClass}">${p.status}</span>
                            </div>
                            <div style="font-size:13px; color:#f59e0b;">Main Violations: ${violations}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:5px;">Required Check: ${required.join(', ') || 'None'}</div>
                            <div class="ppe-grid">
                                ${ppeBox('Helmet', ppe.helmet, required.includes('helmet'), conf.helmet)}
                                ${ppeBox('Vest', ppe.vest, required.includes('vest'), conf.vest)}
                            </div>
                        </div>`;
                    }).join('');
                } else {
                    container.innerHTML = '<div style="color: #64748b; text-align: center; padding: 20px;">No persons detected</div>';
                }

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
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/sensors')
def api_sensors():
    return jsonify(sensor_data)

@app.route('/api/cv')
def api_cv():
    return jsonify(cv_state)


@app.route('/api/mode', methods=['GET', 'POST'])
def api_mode():
    global segment_mode

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        mode = str(data.get('mode', segment_mode)).lower().strip()
        if mode in ['auto', 'head', 'torso', 'full']:
            segment_mode = mode

    return jsonify({"mode": segment_mode})

@app.route('/api/alerts')
def api_alerts():
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

def mqtt_recieve():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    if USERNAME:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    print(f"Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

    client.loop_forever()

# ========================== MAIN ==========================
if __name__ == '__main__':

    threading.Thread(target=mqtt_recieve, daemon=True).start()

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
        print("[INIT] Make sure the model file exists at the path above")
        exit(1)

    print("[INIT] Starting CV thread...")
    threading.Thread(target=cv_thread, daemon=True).start()

    print("[INIT] Starting Sensor thread...")
    #threading.Thread(target=sensor_thread, daemon=True).start()

    print("[INIT] Starting web server...")
    print("[INIT] Dashboard: http://localhost:5000")
    print("[INIT] Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
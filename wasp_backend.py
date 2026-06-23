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

import cv2
import numpy as np
import re
from flask import Flask, render_template_string, Response, jsonify, request
from ultralytics import YOLO
import sqlite3
import threading
import time
import requests
import os
import serial
import json
from datetime import datetime, date, timedelta
from urllib.parse import quote
import torch
import paho.mqtt.client as mqtt
from groq import Groq

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

# YOLOv8 Model Path (newer trained model)
MODEL_PATH = "newbest.pt"
SIMULATE_CV = False   # True = fake CV data, False = real camera YOLO

# Agent Settings
WARNING_COOLDOWN = 30
HEAT_THRESHOLD = 35.0
CONFIDENCE_THRESHOLD = 0.35

# MQTT Broker
BROKER_HOST = "192.168.100.218"   #recommended localhost if same device
BROKER_PORT = 1883          #depends on running device port
TOPIC       = "sensors/#"
USERNAME    = ""            # if available
PASSWORD    = ""            # if available

# Ollama AI Agent (Primary)
OLLAMA_URL = "http://192.168.212.193:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 60  # seconds

# Agent Mode Toggle
AGENT_MODE = "groq"  # "ollama" or "groq"

# Groq AI Agent (Fallback)
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY = "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"

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
    "global_helmet": True,
    "global_vest": True,
    "global_goggles": True,
    "global_gloves": True,
    "global_boots": True,
    "last_update": "N/A"
}

active_warnings = {}
model = None
tts_available = False
simulation_mode = False

# Reusable MQTT client (connected once, reused for publish)
mqtt_client = None
mqtt_client_ready = False
mqtt_client_lock = threading.Lock()

# ========================== MQTT ==========================
def on_connect(client, userdata, flags, reason_code, properties):
    global mqtt_client_ready
    if reason_code == 0:
        print(f"[MQTT] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
        with mqtt_client_lock:
            mqtt_client_ready = True
    else:
        print(f"[MQTT] Connection failed, reason code: {reason_code}")

def on_message(client, userdata, msg):
    global sensor_data
    topic = msg.topic
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
        print(f"[SENSOR] Motion:{motion} Obstacle:{obstacle} Humidity:{humidity} Temp:{temp} Mic:{mic_level} Gas:{mq2_raw}")

    except (json.JSONDecodeError, KeyError) as e:
        print(f"[MQTT] Bad payload: {e}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    global mqtt_client_ready
    with mqtt_client_lock:
        mqtt_client_ready = False
    print(f"[MQTT] Disconnected ({reason_code})")

def mqtt_publish(topic, payload):
    """Reuse the already-connected MQTT client to publish."""
    global mqtt_client, mqtt_client_ready
    with mqtt_client_lock:
        if mqtt_client_ready and mqtt_client is not None:
            try:
                mqtt_client.publish(topic, json.dumps(payload))
                print(f"[MQTT] Published to {topic}: {payload}")
            except Exception as e:
                print(f"[MQTT] Publish error: {e}")
        else:
            print(f"[MQTT] Client not ready. Would publish: {topic} -> {payload}")

# ==============================================================

# ========================== DATABASE ==========================
def init_db():
    # Delete existing DB to ensure fresh schema
    if os.path.exists("wasp.db"):
        try:
            os.remove("wasp.db")
        except Exception as e:
            print(f"[DB] Could not delete old wasp.db: {e}")
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_type TEXT,
            details TEXT,
            status TEXT DEFAULT 'ACTIVE'
        )""")
        c.execute("DROP TABLE IF EXISTS agent_decisions")
        c.execute("""CREATE TABLE agent_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            context_json TEXT,
            decision_json TEXT,
            risk_level TEXT,
            model_used TEXT,
            tool_calls TEXT,
            response_time_ms INTEGER DEFAULT 0
        )""")
        conn.commit()
        conn.close()
        print("[DB] Database initialized (alerts + agent_decisions with correct schema)")
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

def log_agent_decision(risk_level, reasoning, speak_bm, speak_en, action_tier, notify_supervisor, context_dict, log_note,
                       model_used=None, tool_calls=None, response_time_ms=0):
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        decision_json = {
            "risk_level": risk_level,
            "reasoning": reasoning,
            "speak_bm": speak_bm,
            "speak_en": speak_en,
            "action_tier": action_tier,
            "notify_supervisor": bool(notify_supervisor),
            "log_note": log_note
        }
        c.execute("""INSERT INTO agent_decisions 
            (timestamp, context_json, decision_json, risk_level, model_used, tool_calls, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             json.dumps(context_dict),
             json.dumps(decision_json),
             risk_level,
             model_used,
             json.dumps(tool_calls) if tool_calls else None,
             response_time_ms))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB Agent Log Error] {e}")

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

# ========================== FLUTTER NOTIFICATION PLACEHOLDER ==========================
def flutter_notify(payload):
    """
    Placeholder for Flutter push notification integration (Phase 3).
    Prints the full payload as labelled JSON so it's easy to parse/forward.
    """
    print("[FLUTTER READY] " + json.dumps(payload))

# ========================== SPATIAL MATCHING (Born) ==========================
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

# ========================== AGENT TOOLS ==========================
def get_sensor_trend(minutes: int = 5):
    """Query SQLite for avg temp/humidity/gas over last N minutes."""
    print(f"[TOOL CALL] get_sensor_trend(minutes={minutes})")
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        since = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        # sensor_readings table doesn't exist, use current sensor_data
        # For trend we'd ideally have a history table, but return current snapshot
        result = {
            "avg_temp": sensor_data.get("temperature", 0.0),
            "avg_humidity": sensor_data.get("humidity", 0.0),
            "avg_gas": sensor_data.get("air_quality", 0),
            "minutes": minutes,
            "note": "current snapshot (no historical sensor table yet)"
        }
        conn.close()
        print(f"[TOOL RESULT] {result}")
        return result
    except Exception as e:
        err = {"error": str(e)}
        print(f"[TOOL RESULT] {err}")
        return err

def get_violation_history(minutes: int = 10):
    """Return violation counts per PPE type from SQLite agent_decisions."""
    print(f"[TOOL CALL] get_violation_history(minutes={minutes})")
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        since = (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
        # Query alerts for PPE violations
        c.execute("""SELECT alert_type, COUNT(*) FROM alerts 
                     WHERE timestamp >= ? AND alert_type LIKE '%VIOLATION%' 
                     GROUP BY alert_type""", (since,))
        rows = c.fetchall()
        conn.close()
        result = {row[0]: row[1] for row in rows} if rows else {}
        print(f"[TOOL RESULT] {result}")
        return result
    except Exception as e:
        err = {"error": str(e)}
        print(f"[TOOL RESULT] {err}")
        return err

def get_worker_count():
    """Return current person_count from cv_state."""
    print("[TOOL CALL] get_worker_count()")
    count = cv_state.get("person_count", 0)
    result = {"person_count": count}
    print(f"[TOOL RESULT] {result}")
    return result

def trigger_esp32_alert():
    """Publish ALERT to MQTT topic wasp/alert."""
    print("[TOOL CALL] trigger_esp32_alert()")
    mqtt_publish("wasp/alert", "ALERT")
    result = {"status": "sent", "topic": "wasp/alert"}
    print(f"[TOOL RESULT] {result}")
    return result

def push_supervisor_alert(message: str, risk_level: str):
    """Print [FLUTTER READY] JSON for supervisor notification."""
    print(f"[TOOL CALL] push_supervisor_alert(message='{message}', risk_level='{risk_level}')")
    payload = {
        "type": "SUPERVISOR_ALERT",
        "risk_level": risk_level,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "person_count": cv_state.get("person_count", 0),
        "ppe_status": {
            "helmet": cv_state.get("global_helmet", True),
            "vest": cv_state.get("global_vest", True)
        }
    }
    flutter_notify(payload)
    result = {"status": "notified", "risk_level": risk_level}
    print(f"[TOOL RESULT] {result}")
    return result

def log_incident(severity: str, description: str):
    """Write incident to SQLite agent_decisions."""
    print(f"[TOOL CALL] log_incident(severity='{severity}', description='{description}')")
    try:
        conn = sqlite3.connect("wasp.db", check_same_thread=False)
        c = conn.cursor()
        decision_json = {
            "risk_level": severity,
            "reasoning": description,
            "speak_bm": "",
            "speak_en": "",
            "action_tier": 1,
            "notify_supervisor": False,
            "log_note": description
        }
        c.execute("""INSERT INTO agent_decisions 
            (timestamp, context_json, decision_json, risk_level, model_used, tool_calls, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "{}",
             json.dumps(decision_json),
             severity,
             None,
             None,
             0))
        conn.commit()
        conn.close()
        result = {"status": "logged", "severity": severity}
        print(f"[TOOL RESULT] {result}")
        return result
    except Exception as e:
        err = {"error": str(e)}
        print(f"[TOOL RESULT] {err}")
        return err

# Tool definitions for Ollama function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_sensor_trend",
            "description": "Get average sensor readings (temp, humidity, gas) over the last N minutes",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "Time window in minutes", "default": 5}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_violation_history",
            "description": "Get violation counts per PPE type over the last N minutes",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {"type": "integer", "description": "Time window in minutes", "default": 10}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_worker_count",
            "description": "Get current number of workers detected on site",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_esp32_alert",
            "description": "Send immediate alert to ESP32 via MQTT to activate buzzer/LED",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "push_supervisor_alert",
            "description": "Notify site supervisor via Flutter push notification",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Alert message"},
                    "risk_level": {"type": "string", "description": "LOW|MEDIUM|HIGH|CRITICAL"}
                },
                "required": ["message", "risk_level"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_incident",
            "description": "Log an incident to the agent_decisions database",
            "parameters": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "description": "LOW|MEDIUM|HIGH|CRITICAL"},
                    "description": {"type": "string", "description": "Incident description"}
                },
                "required": ["severity", "description"]
            }
        }
    }
]

# Map tool names to functions
TOOL_MAP = {
    "get_sensor_trend": get_sensor_trend,
    "get_violation_history": get_violation_history,
    "get_worker_count": get_worker_count,
    "trigger_esp32_alert": trigger_esp32_alert,
    "push_supervisor_alert": push_supervisor_alert,
    "log_incident": log_incident
}

# ========================== AI AGENT ==========================
class WASPAgent:
    """
    WASP AI reasoning agent with Ollama (primary) + Groq (fallback).
    Supports tool calling / function calling for agentic loop.
    """

    def __init__(self):
        self.groq_client = None
        self.model_active = "none"
        self.last_response_time_ms = 0
        self.last_tool_calls = []
        self.groq_cooldown = 15
        self.last_groq_call = 0
        self.last_report_date = None
        self.model = GROQ_MODEL

        # Init Groq client for fallback
        api_key = GROQ_API_KEY
        if api_key and api_key != "your_actual_groq_api_key_here":
            try:
                self.groq_client = Groq(api_key=api_key)
                print(f"[WASPAgent] Groq fallback client ready (model: {GROQ_MODEL})")
            except Exception as e:
                print(f"[WASPAgent] Failed to init Groq: {e}")

    def _call_ollama(self, messages, tools=None, tool_choice="auto"):
        """Call Ollama /api/generate with a single prompt string."""
        url = f"{OLLAMA_URL}/api/generate"
        # Convert messages list to a single prompt string
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"[ASSISTANT]\n{content}")
            elif role == "tool":
                prompt_parts.append(f"[TOOL]\n{content}")
        prompt = "\n".join(prompt_parts)
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        # Note: /api/generate may not support tools; include only if present
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        start = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = int((time.time() - start) * 1000)
            self.last_response_time_ms = elapsed_ms
            self.model_active = "ollama"
            print(f"[Ollama] Response in {elapsed_ms}ms")
            # Wrap response to match expected structure: {"message": {"content": "..."}}
            response_text = data.get("response", "")
            return {
                "message": {
                    "role": "assistant",
                    "content": response_text,
                    "tool_calls": None
                }
            }
        except requests.exceptions.Timeout:
            print(f"[WASPAgent] Ollama timeout after {OLLAMA_TIMEOUT}s")
            return None
        except Exception as e:
            print(f"[WASPAgent] Ollama error detail: {type(e).__name__}: {e}")
            return None

    def _call_groq(self, messages, tools=None, tool_choice="auto"):
        """Call Groq chat completions with optional tools."""
        if self.groq_client is None:
            return None
        try:
            start = time.time()
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 512
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            response = self.groq_client.chat.completions.create(**kwargs)
            elapsed_ms = int((time.time() - start) * 1000)
            self.last_response_time_ms = elapsed_ms
            self.model_active = "groq"
            # Convert to dict for consistency
            msg = response.choices[0].message
            return {
                "message": {
                    "role": msg.role,
                    "content": msg.content,
                    "tool_calls": getattr(msg, "tool_calls", None)
                }
            }
        except Exception as e:
            print(f"[WASPAgent] Groq API error: {e}")
            return None

    def analyze(self, context):
        """
        Agentic loop:
        1. Build initial prompt with tools
        2. Call selected provider (Ollama or Groq) based on AGENT_MODE
        3. If tool_calls -> execute tools, print [TOOL CALL]/[RESULT]
        4. Feed results back for final decision
        5. Print [AGENT DECISION]
        """
        print(f"[WASPAgent] Mode: {AGENT_MODE}")
        now = time.time()
        if now - self.last_groq_call < self.groq_cooldown:
            print("[WASPAgent] Cooldown active, using rule-based fallback")
            return self._rule_based_fallback(context)

        severity = self._get_severity(context)
        self.last_tool_calls = []

        system_prompt = (
            "You are WASP, an autonomous construction site safety officer. "
            "You monitor workers using IoT sensors and computer vision. "
            "You have access to tools to gather information. "
            "Use tools when needed to make informed decisions. "
            "After gathering information, provide a final decision. "
            "You respond ONLY in valid JSON with this exact structure:\n"
            "{\n"
            "  'risk_level': 'LOW|MEDIUM|HIGH|CRITICAL',\n"
            "  'reasoning': 'brief explanation in English',\n"
            "  'speak_bm': 'what to say in Bahasa Malaysia (only if HIGH or CRITICAL)',\n"
            "  'speak_en': 'what to say in English (only if HIGH or CRITICAL)',\n"
            "  'notify_supervisor': true/false,\n"
            "  'action_tier': 1-4,\n"
            "  'log_note': 'one line for incident log'\n"
            "}"
        )

        user_message = json.dumps(context, indent=2)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current site conditions:\n{user_message}"}
        ]

        # Call selected provider only
        if AGENT_MODE == "ollama":
            print(f"[WASPAgent] Calling Ollama at {OLLAMA_URL} with model {OLLAMA_MODEL}")
            llm_response = self._call_ollama(messages, tools=TOOL_DEFINITIONS)
            if llm_response is None:
                print("[WASPAgent] Ollama failed, using rule-based fallback")
                return self._rule_based_fallback(context)
        elif AGENT_MODE == "groq":
            llm_response = self._call_groq(messages, tools=TOOL_DEFINITIONS)
            if llm_response is None:
                print("[WASPAgent] Groq failed, using rule-based fallback")
                return self._rule_based_fallback(context)
        else:
            llm_response = None

        if llm_response is None:
            print("[WASPAgent] Provider failed, using rule-based fallback")
            return self._rule_based_fallback(context)

        self.last_groq_call = time.time()

        # Process tool calls if any
        msg = llm_response.get("message", {})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            # Assistant wants to call tools
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": tool_calls})

            for tc in tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)

                self.last_tool_calls.append({"name": fn_name, "args": fn_args})

                if fn_name in TOOL_MAP:
                    result = TOOL_MAP[fn_name](**(fn_args if fn_args is not None else {}))
                else:
                    result = {"error": f"Unknown tool {fn_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(tc, "id", ""),
                    "content": json.dumps(result)
                })

            # Final call after tools - use same provider only
            if AGENT_MODE == "ollama":
                final_resp = self._call_ollama(messages, tools=None)
            else:
                final_resp = self._call_groq(messages, tools=None)

            if final_resp is None:
                print(f"[WASPAgent] {AGENT_MODE} failed after tools, using rule-based fallback")
                return self._rule_based_fallback(context)

            final_msg = final_resp.get("message", {})
            raw = final_msg.get("content", "")
        else:
            raw = msg.get("content", "")

        # Parse decision JSON
        raw = raw.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            raw = match.group()
        raw = raw.replace("'", '"')
        try:
            decision = json.loads(raw)
        except Exception as e:
            print(f"[WASPAgent] JSON parse error: {e}")
            return self._rule_based_fallback(context)

        decision["model_used"] = self.model_active
        print(f"[AGENT DECISION] {decision.get('risk_level', 'UNKNOWN')} | tier {decision.get('action_tier', '-')} | {decision.get('reasoning', '')[:120]}")
        return decision

    def _build_context(self):
        """Build site context dict from sensor_data and cv_state globals."""
        temp = sensor_data.get("temperature", 0.0)
        gas = sensor_data.get("gas_level", sensor_data.get("air_quality", 0))
        humidity = sensor_data.get("humidity", 0.0)
        person_count = cv_state.get("person_count", 0)
        worker_detected = person_count > 0
        # Derive global PPE flags from cv_state
        helmet = cv_state.get("global_helmet", True)
        vest = cv_state.get("global_vest", True)
        gloves = cv_state.get("global_gloves", True)
        boots = cv_state.get("global_boots", True)
        goggles = cv_state.get("global_goggles", True)
        # If per-person ppe exists, AND logic across persons
        persons = cv_state.get("persons", [])
        if persons:
            helmet = all(p.get("ppe", {}).get("helmet", False) for p in persons)
            vest = all(p.get("ppe", {}).get("vest", False) for p in persons)
            gloves = all(p.get("ppe", {}).get("gloves", False) for p in persons)
            boots = all(p.get("ppe", {}).get("boots", False) for p in persons)
            goggles = all(p.get("ppe", {}).get("goggles", False) for p in persons)
        return {
            "environment": {"temperature": temp, "gas_level": gas, "humidity": humidity},
            "ppe_status": {"helmet": helmet, "vest": vest, "gloves": gloves, "boots": boots, "goggles": goggles},
            "motion": {"person_count": person_count, "worker_detected": worker_detected},
            "timestamp": datetime.now().isoformat()
        }

    def _get_severity(self, context):
        """Local rule-based severity check."""
        temp = context["environment"]["temperature"]
        gas = context["environment"]["gas_level"]
        ppe = context["ppe_status"]

        if temp > HEAT_THRESHOLD and gas > 500:
            return "CRITICAL"
        if temp > HEAT_THRESHOLD:
            return "HIGH"
        if not ppe["helmet"] or not ppe["vest"]:
            return "HIGH"
        if not ppe["gloves"] or not ppe["boots"] or not ppe["goggles"]:
            return "MEDIUM"
        if 30 < temp <= HEAT_THRESHOLD:
            return "MEDIUM"
        if context["motion"]["worker_detected"] and gas > 300:
            return "MEDIUM"
        return "LOW"

    def _rule_based_fallback(self, context):
        """Fallback decision when both LLMs unavailable."""
        severity = self._get_severity(context)
        temp = context["environment"]["temperature"]
        ppe = context["ppe_status"]
        person_count = context["motion"]["person_count"]
        self.model_active = "rule-based"

        if severity == "CRITICAL":
            return {
                "risk_level": "CRITICAL",
                "reasoning": f"Extreme heat ({temp:.1f}C) with hazardous gas levels.",
                "speak_bm": "BAHAYA! Suhu melampau dan gas berbahaya. Sila keluar kawasan!",
                "speak_en": "DANGER! Extreme heat and gas hazard. Evacuate immediately!",
                "notify_supervisor": True, "action_tier": 4,
                "log_note": f"CRITICAL: Heat {temp:.1f}C + gas.",
                "model_used": "rule-based"
            }
        elif severity == "HIGH":
            missing = [k for k in ["helmet", "vest"] if not ppe[k]]
            if temp > HEAT_THRESHOLD:
                missing.append(f"heat ({temp:.1f}C)")
            return {
                "risk_level": "HIGH",
                "reasoning": f"Critical PPE missing: {missing}.",
                "speak_bm": f"Perhatian! {', '.join(missing)} tidak dipakai!",
                "speak_en": f"Warning! {', '.join(missing)} not worn!",
                "notify_supervisor": True, "action_tier": 3,
                "log_note": f"HIGH: Missing {missing}.",
                "model_used": "rule-based"
            }
        elif severity == "MEDIUM":
            missing = [k for k in ["gloves", "boots", "goggles"] if not ppe[k]]
            return {
                "risk_level": "MEDIUM",
                "reasoning": f"Non-critical PPE missing: {missing}.",
                "speak_bm": f"Ingatan! {', '.join(missing)} tidak dipakai.",
                "speak_en": f"Reminder! {', '.join(missing)} not detected.",
                "notify_supervisor": False, "action_tier": 2,
                "log_note": f"MEDIUM: Missing {missing}.",
                "model_used": "rule-based"
            }
        else:
            return {
                "risk_level": "LOW", "reasoning": "All clear.",
                "speak_bm": "", "speak_en": "",
                "notify_supervisor": False, "action_tier": 1,
                "log_note": "LOW: Site safe.",
                "model_used": "rule-based"
            }


# ========================== DECISION ENGINE ==========================
# Global agent instance (initialized in main)
wasp_agent = None

def agent_engine():
    """
    Agentic decision engine.
    - Builds context from CV + sensor data
    - For HIGH/CRITICAL: uses WASPAgent.analyze() which calls tools + LLM
    - For LOW/MEDIUM: silent log or rule-based
    """
    global active_warnings
    current_time = time.time()
    context = wasp_agent._build_context()
    severity = wasp_agent._get_severity(context)

    # LOW / MEDIUM: lightweight handling
    if severity in ("LOW", "MEDIUM"):
        if severity == "MEDIUM":
            decision = wasp_agent._rule_based_fallback(context)
            log_alert("AGENT_MEDIUM", decision["log_note"])
            if decision["speak_bm"]:
                speak(decision["speak_bm"])
            log_agent_decision(
                decision["risk_level"], decision["reasoning"],
                decision["speak_bm"], decision["speak_en"],
                decision["action_tier"], decision["notify_supervisor"],
                context, decision["log_note"]
            )
        for k in list(active_warnings.keys()):
            if k.startswith("CRIT_") or k.startswith("HIGH_"):
                del active_warnings[k]
        return

    # HIGH / CRITICAL: agentic analysis
    if severity == "HIGH":
        v_key = "HIGH_ppe"
        ppe = context["ppe_status"]
        missing_parts = []
        if not ppe["helmet"]:
            missing_parts.append("no_helmet")
        if not ppe["vest"]:
            missing_parts.append("no_vest")
        if context["environment"]["temperature"] > HEAT_THRESHOLD:
            missing_parts.append("heat")
        if missing_parts:
            v_key = "HIGH_" + "_".join(missing_parts)
    elif severity == "CRITICAL":
        v_key = "CRIT_heat_gas"
    else:
        v_key = f"{severity}_{int(current_time)}"

    if v_key in active_warnings:
        elapsed = current_time - active_warnings[v_key]
        if elapsed < WARNING_COOLDOWN:
            return

    active_warnings[v_key] = current_time

    def run_agent():
        decision = wasp_agent.analyze(context)
        if decision is None:
            return

        model_used = wasp_agent.model_active
        tool_calls = wasp_agent.last_tool_calls
        response_ms = wasp_agent.last_response_time_ms

        log_agent_decision(
            decision.get("risk_level", severity),
            decision.get("reasoning", ""),
            decision.get("speak_bm", ""),
            decision.get("speak_en", ""),
            decision.get("action_tier", 1),
            decision.get("notify_supervisor", False),
            context,
            decision.get("log_note", ""),
            model_used=model_used,
            tool_calls=tool_calls,
            response_time_ms=response_ms
        )

        action_tier = decision.get("action_tier", 2)

        if action_tier >= 2:
            bm_msg = decision.get("speak_bm", "")
            if bm_msg:
                speak(bm_msg)
            en_msg = decision.get("speak_en", "")
            log_alert(f"AGENT_{decision.get('risk_level', severity)}",
                      f"{en_msg} | {decision.get('log_note', '')}")

        if action_tier >= 3:
            flutter_payload = {
                "type": "WASP_ALERT",
                "risk_level": decision.get("risk_level", severity),
                "timestamp": datetime.now().isoformat(),
                "speak_en": decision.get("speak_en", ""),
                "speak_bm": decision.get("speak_bm", ""),
                "reasoning": decision.get("reasoning", ""),
                "log_note": decision.get("log_note", ""),
                "action_tier": action_tier,
                "person_count": context["motion"]["person_count"],
                "temperature": context["environment"]["temperature"],
                "ppe_status": context["ppe_status"],
                "model_used": model_used,
                "tool_calls": tool_calls,
                "response_time_ms": response_ms
            }
            flutter_notify(flutter_payload)

        if action_tier >= 4:
            mqtt_publish("esp32/command", {
                "cmd": "alarm",
                "severity": decision.get("risk_level", severity),
                "duration": 5,
                "reason": decision.get("log_note", "")
            })

    threading.Thread(target=run_agent, daemon=True).start()


# ========================== CV THREAD (Born logic) ==========================
def cv_thread():
    global latest_frame, cv_state

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
                            "person_id": "person_1",
                            "bbox": [120, 70, 420, 480],
                            "confidence": 0.96,
                            "ppe": {
                                "helmet": True,
                                "vest": True,
                                "goggles": True,
                                "gloves": True,
                                "boots": True,
                                "no_helmet": False,
                                "no_goggle": False,
                                "no_gloves": False,
                                "no_boots": False
                            },
                            "violations": [],
                            "status": "SAFE"
                        }
                    ],
                    "any_violation": False,
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
                            "person_id": "person_1",
                            "bbox": [120, 70, 420, 480],
                            "confidence": 0.94,
                            "ppe": {
                                "helmet": False,
                                "vest": True,
                                "goggles": False,
                                "gloves": False,
                                "boots": True,
                                "no_helmet": True,
                                "no_goggle": True,
                                "no_gloves": True,
                                "no_boots": False
                            },
                            "violations": [
                                "Missing helmet",
                                "Missing goggles",
                                "Missing gloves"
                            ],
                            "status": "VIOLATION"
                        }
                    ],
                    "any_violation": True,
                    "global_helmet": False,
                    "global_vest": True,
                    "global_goggles": False,
                    "global_gloves": False,
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
                imgsz=416
            )

            annotated = results[0].plot()
            print(f"[CV] Frame shape: {annotated.shape}, has boxes: {len(results[0].boxes)}")

            persons = []
            ppe_items = []

            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = results[0].names[cls].lower()

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [round(x1), round(y1), round(x2), round(y2)]

                detection = {
                    "class_name": name,
                    "confidence": round(conf, 2),
                    "bbox": bbox
                }

                if name == "person":
                    persons.append(detection)
                else:
                    ppe_items.append(detection)



            person_outputs = []

            for idx, person in enumerate(persons, start=1):
                p_box = person["bbox"]

                ppe_status = {
                    "helmet": False,
                    "vest": False,
                    "goggles": False,
                    "gloves": False,
                    "boots": False,
                    "no_helmet": False,
                    "no_goggle": False,
                    "no_gloves": False,
                    "no_boots": False
                }

                for item in ppe_items:
                    item_name = item["class_name"]
                    item_box = item["bbox"]

                    if is_inside(item_box, p_box):
                        if item_name in ppe_status:
                            ppe_status[item_name] = True

                violations = []

                # Strict demo rule: helmet and vest only
                if ppe_status["no_helmet"] or not ppe_status["helmet"]:
                    violations.append("Missing helmet")

                if not ppe_status["vest"]:
                    violations.append("Missing vest")

                status = "SAFE" if len(violations) == 0 else "VIOLATION"

                person_outputs.append({
                    "person_id": f"person_{idx}",
                    "bbox": p_box,
                    "confidence": person["confidence"],
                    "ppe": ppe_status,
                    "violations": violations,
                    "status": status
                })

          # --- Draw metrics overlay ---
            current_fps = fps_counter / max(time.time() - fps_start, 0.001)
            lines = [
                ("CUDA" if CUDA_AVAILABLE else "CPU", (0, 255, 0) if CUDA_AVAILABLE else (0, 255, 255)),
                (f"FPS: {current_fps:.1f}", (255, 255, 255)),
                (f"Persons: {len(person_outputs)}", (255, 255, 255)),
                (f"Violations: {sum(1 for p in person_outputs if p['status'] == 'VIOLATION')}", (255, 150, 150)),
                (f"Conf: {CONFIDENCE_THRESHOLD}", (200, 200, 255)),
                (f"Agent: {AGENT_MODE.upper()}", (255, 255, 255)),
                (f"Model: {OLLAMA_MODEL if AGENT_MODE == 'ollama' else GROQ_MODEL}", (255, 255, 255))
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 1
            x = 10
            y_start = 30
            line_h = 22
            max_w = 0
            for txt, _ in lines:
                (tw, _), _ = cv2.getTextSize(txt, font, scale, thick)
                if tw > max_w:
                    max_w = tw
            pad = 6
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x - pad, y_start - 20), (x - pad + max_w + pad * 2, y_start - 20 + len(lines) * line_h + pad), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)
            for i, (txt, color) in enumerate(lines):
                y = y_start + i * line_h
                cv2.putText(annotated, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)
            # --- End overlay ---

            any_violation = any(p["status"] == "VIOLATION" for p in person_outputs)

            g_helmet = all(p["ppe"]["helmet"] for p in person_outputs) if person_outputs else True
            g_vest = all(p["ppe"]["vest"] for p in person_outputs) if person_outputs else True
            g_goggles = all(p["ppe"]["goggles"] for p in person_outputs) if person_outputs else True
            g_gloves = all(p["ppe"]["gloves"] for p in person_outputs) if person_outputs else True
            g_boots = all(p["ppe"]["boots"] for p in person_outputs) if person_outputs else True

            cv_state = {
                "person_count": len(person_outputs),
                "persons": person_outputs,
                "any_violation": any_violation,
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

# ========================== DAILY REPORT THREAD ==========================
def daily_report_loop():
    """
    Fires at 18:00 (6PM) daily.
    Generates TWO files in /reports folder:
      1. daily_report_YYYY-MM-DD.json — raw aggregated data from SQLite
      2. daily_report_YYYY-MM-DD.txt — Groq-generated OSHA-style narrative
    """
    global wasp_agent

    os.makedirs("reports", exist_ok=True)

    while True:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # Skip if report already generated today
        if wasp_agent and wasp_agent.last_report_date == today_str:
            time.sleep(60)
            continue

        # Check if it's 18:00 (or past 18:00 and not yet run today)
        if now.hour >= 18:
            today_start = f"{today_str} 00:00:00"
            tomorrow_start = (datetime.strptime(today_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"

            try:
                conn = sqlite3.connect("wasp.db", check_same_thread=False)
                c = conn.cursor()

                # --- Aggregate alert data ---
                c.execute("SELECT COUNT(*) FROM alerts WHERE timestamp >= ? AND timestamp < ?",
                          (today_start, tomorrow_start))
                total_alerts = c.fetchone()[0]

                c.execute("SELECT alert_type, COUNT(*) FROM alerts WHERE timestamp >= ? AND timestamp < ? GROUP BY alert_type",
                          (today_start, tomorrow_start))
                alerts_by_type = {row[0]: row[1] for row in c.fetchall()}

                # --- Aggregate agent decisions ---
                c.execute("SELECT COUNT(*) FROM agent_decisions WHERE timestamp >= ? AND timestamp < ?",
                          (today_start, tomorrow_start))
                total_decisions = c.fetchone()[0]

                c.execute("SELECT risk_level, COUNT(*) FROM agent_decisions WHERE timestamp >= ? AND timestamp < ? GROUP BY risk_level",
                          (today_start, tomorrow_start))
                decisions_by_risk = {row[0]: row[1] for row in c.fetchall()}

                c.execute("SELECT action_tier, COUNT(*) FROM agent_decisions WHERE timestamp >= ? AND timestamp < ? GROUP BY action_tier",
                          (today_start, tomorrow_start))
                actions_by_tier = {f"tier_{row[0]}": row[1] for row in c.fetchall()}

                # --- Full alert list for Groq narrative ---
                c.execute("SELECT timestamp, alert_type, details FROM alerts WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
                          (today_start, tomorrow_start))
                alert_rows = [{"time": r[0], "type": r[1], "details": r[2]} for r in c.fetchall()]

                c.execute("SELECT timestamp, risk_level, reasoning, action_tier FROM agent_decisions WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
                          (today_start, tomorrow_start))
                decision_rows = [{"time": r[0], "risk": r[1], "reason": r[2], "tier": r[3]} for r in c.fetchall()]

                conn.close()

                # --- Build report JSON ---
                report = {
                    "date": today_str,
                    "total_alerts": total_alerts,
                    "total_agent_decisions": total_decisions,
                    "alerts_by_type": alerts_by_type,
                    "decisions_by_risk_level": decisions_by_risk,
                    "actions_by_tier": actions_by_tier,
                    "peak_temperature": max(sensor_data.get("temperature", 0), 0),
                    "alerts": alert_rows,
                    "agent_decisions": decision_rows
                }

                # Write JSON report
                json_path = f"reports/daily_report_{today_str}.json"
                with open(json_path, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"[REPORT] JSON saved: {json_path}")

                # Generate Groq narrative for the TXT report
                txt_path = f"reports/daily_report_{today_str}.txt"
                narrative = _generate_narrative_report(report)
                with open(txt_path, "w") as f:
                    f.write(narrative)
                print(f"[REPORT] Narrative saved: {txt_path}")

                # Mark today as done
                if wasp_agent:
                    wasp_agent.last_report_date = today_str

            except Exception as e:
                print(f"[REPORT Error] {e}")

        time.sleep(60)


def _generate_narrative_report(report_data):
    """Generate OSHA-style narrative via Groq, or fallback text."""
    agent = wasp_agent
    if agent and agent.groq_client:
        system_prompt = (
            "You are WASP, an autonomous construction site safety officer. "
            "Write an official end-of-day OSHA-style safety report in English summarizing today's incidents. "
            "Reference actual counts, times, and risk levels from the data provided. "
            "Be professional, factual, and concise. "
            "If there were no incidents, state that the site was safe and compliant today."
        )
        try:
            user_message = json.dumps(report_data, indent=2)
            response = agent.groq_client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Today's site data:\n{user_message}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            print(f"[REPORT] Groq narrative error: {e}")

    # Fallback if Groq unavailable
    date_str = report_data.get("date", "unknown")
    total = report_data.get("total_alerts", 0)
    decisions = report_data.get("total_agent_decisions", 0)
    lines = [
        f"WASP Daily Safety Report — {date_str}",
        "=" * 50,
        "",
        f"Total Alerts Today: {total}",
        f"Agent Decisions: {decisions}",
        "",
        "Alerts by Type:",
    ]
    for atype, count in report_data.get("alerts_by_type", {}).items():
        lines.append(f"  - {atype}: {count}")
    lines.extend([
        "",
        "Risk Level Breakdown:",
    ])
    for risk, count in report_data.get("decisions_by_risk_level", {}).items():
        lines.append(f"  - {risk}: {count}")
    lines.extend([
        "",
        f"Peak Temperature: {report_data.get('peak_temperature', 0):.1f}C",
        "",
        "--- End of Report ---"
    ])
    return "\n".join(lines)


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
        .badge-safe { background: #22c55e; color: #0f172a; }
        .person-card { background: #0f172a; padding: 15px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #334155; }
        .person-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .person-id { font-weight: bold; color: #38bdf8; }
        .person-status { font-size: 12px; font-weight: bold; padding: 2px 8px; border-radius: 8px; }
        .person-status.safe { background: #22c55e; color: #0f172a; }
        .person-status.violation { background: #ef4444; color: white; }
        .person-violations { font-size: 13px; color: #f59e0b; }
        .ppe-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-top: 10px; }
        .ppe-item { text-align: center; padding: 8px; border-radius: 6px; font-size: 11px; }
        .ppe-ok { background: #22c55e33; color: #22c55e; border: 1px solid #22c55e; }
        .ppe-missing { background: #ef444433; color: #ef4444; border: 1px solid #ef4444; }
        .footer { text-align: center; padding: 20px; color: #64748b; font-size: 12px; border-top: 1px solid #334155; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .agent-risk-low { color: #22c55e; font-weight: bold; }
        .agent-risk-medium { color: #f59e0b; font-weight: bold; }
        .agent-risk-high { color: #f97316; font-weight: bold; }
        .agent-risk-critical { color: #ef4444; font-weight: bold; }
        .agent-reasoning { font-size: 13px; color: #cbd5e1; margin: 8px 0; }
        .agent-tier { font-size: 11px; color: #94a3b8; }
    </style>
</head>
<body>
    <div class="header">WASP <span>Warden Autonomous Safety Platform</span></div>
    <div id="alert-banner" class="alert-banner">VIOLATION DETECTED - SPEAKING WARNING</div>
    <div class="grid">
        <div class="panel video-container">
            <h3>Live Feed - Zone A</h3>
            <img src="/video_feed" alt="Camera Feed" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22400%22 height=%22300%22><rect fill=%22%23334155%22 width=%22400%22 height=%22300%22/><text fill=%22%2394a3b8%22 x=%2250%%22 y=%2250%%22 text-anchor=%22middle%22>Camera Offline</text></svg>'">
        </div>
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="panel">
                <h3>Environmental Sensors</h3>
                <div class="sensor-grid">
                    <div class="sensor-box"><div class="sensor-label">Temperature</div><div class="sensor-value" id="temp">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Humidity</div><div class="sensor-value" id="humid">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Motion</div><div class="sensor-value" id="motion">--</div></div>
                    <div class="sensor-box"><div class="sensor-label">Status</div><div class="sensor-value" id="status">--</div></div>
                </div>
            </div>
            <div class="panel">
                <h3>Global PPE Status</h3>
                <div class="status-row"><span class="status-label">Helmet</span><span id="g-helmet" class="status-safe">--</span></div>
                <div class="status-row"><span class="status-label">Vest</span><span id="g-vest" class="status-safe">--</span></div>
                <div class="status-row"><span class="status-label">Goggles</span><span id="g-goggles" class="status-safe">--</span></div>
                <div class="status-row"><span class="status-label">Gloves</span><span id="g-gloves" class="status-safe">--</span></div>
                <div class="status-row" style="border-bottom: none;"><span class="status-label">Boots</span><span id="g-boots" class="status-safe">--</span></div>
            </div>
        </div>
        <div class="panel">
            <h3>AI Agent Status <span id="agent-mode-badge" class="badge" style="background: #38bdf8; color: #0f172a; margin-left: 8px;">GROQ</span></h3>
            <div style="margin-bottom: 12px;">
                <button id="btn-ollama" onclick="setAgentMode('ollama')" style="background: #334155; color: #e2e8f0; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; margin-right: 8px;">Local (Ollama)</button>
                <button id="btn-groq" onclick="setAgentMode('groq')" style="background: #22c55e; color: #0f172a; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px;">Cloud (Groq)</button>
            </div>
            <div id="agent-panel">
                <div style="color: #64748b; text-align: center; padding: 20px;">Waiting for agent decision...</div>
            </div>
        </div>
    </div>
    <div style="max-width: 1400px; margin: 0 auto; padding: 0 20px 20px;">
        <div class="panel">
            <h3>Per-Person Detection (<span id="person-count">0</span> persons)</h3>
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
    <div class="footer">WASP MVP - UTM FAI Showcase 2026 | Running in <span id="mode">Live</span> Mode</div>
    <script>
        async function updateData() {
            try {
                const s = await fetch('/api/sensors').then(r => r.json());
                document.getElementById('temp').textContent = (s.temperature || 0).toFixed(1) + ' C';
                document.getElementById('humid').textContent = (s.humidity || 0).toFixed(1) + '%';
                document.getElementById('motion').textContent = s.motion ? 'DETECTED' : 'CLEAR';
                document.getElementById('motion').style.color = s.motion ? '#f59e0b' : '#22c55e';
                const c = await fetch('/api/cv').then(r => r.json());
                document.getElementById('person-count').textContent = c.person_count || 0;
                const gHelmet = document.getElementById('g-helmet');
                gHelmet.textContent = c.global_helmet ? 'YES' : 'NO';
                gHelmet.className = c.global_helmet ? 'status-safe' : 'status-danger';
                const gVest = document.getElementById('g-vest');
                gVest.textContent = c.global_vest ? 'YES' : 'NO';
                gVest.className = c.global_vest ? 'status-safe' : 'status-danger';
                const gGoggles = document.getElementById('g-goggles');
                gGoggles.textContent = c.global_goggles ? 'YES' : 'NO';
                gGoggles.className = c.global_goggles ? 'status-safe' : 'status-danger';
                const gGloves = document.getElementById('g-gloves');
                gGloves.textContent = c.global_gloves ? 'YES' : 'NO';
                gGloves.className = c.global_gloves ? 'status-safe' : 'status-danger';
                const gBoots = document.getElementById('g-boots');
                gBoots.textContent = c.global_boots ? 'YES' : 'NO';
                gBoots.className = c.global_boots ? 'status-safe' : 'status-danger';
                const hasViolation = c.any_violation;
                const heatStress = (s.temperature || 0) > 35.0;
                const statusEl = document.getElementById('status');
                if (heatStress) { statusEl.textContent = 'HEAT'; statusEl.style.color = '#dc2626'; }
                else if (hasViolation) { statusEl.textContent = 'WARN'; statusEl.style.color = '#f59e0b'; }
                else { statusEl.textContent = 'SAFE'; statusEl.style.color = '#22c55e'; }
                document.getElementById('alert-banner').style.display = (hasViolation || heatStress) ? 'block' : 'none';
                // Render per-person cards
                const container = document.getElementById('persons-container');
                if (c.persons && c.persons.length > 0) {
                    container.innerHTML = c.persons.map(p => {
                        const isSafe = p.status === 'SAFE';
                        const statusClass = isSafe ? 'safe' : 'violation';
                        const statusText = isSafe ? 'SAFE' : 'VIOLATION';
                        const violations = p.violations && p.violations.length > 0 ? p.violations.join(', ') : 'None';
                        const ppe = p.ppe || {};
                        return `<div class="person-card">
                            <div class="person-header">
                                <span class="person-id">${p.person_id}</span>
                                <span class="person-status ${statusClass}">${statusText}</span>
                            </div>
                            <div class="person-violations">Violations: ${violations}</div>
                            <div class="ppe-grid">
                                <div class="ppe-item ${ppe.helmet ? 'ppe-ok' : 'ppe-missing'}">Helmet: ${ppe.helmet ? 'OK' : 'MISS'}</div>
                                <div class="ppe-item ${ppe.vest ? 'ppe-ok' : 'ppe-missing'}">Vest: ${ppe.vest ? 'OK' : 'MISS'}</div>
                                <div class="ppe-item ${ppe.goggles ? 'ppe-ok' : 'ppe-missing'}">Goggles: ${ppe.goggles ? 'OK' : 'MISS'}</div>
                                <div class="ppe-item ${ppe.gloves ? 'ppe-ok' : 'ppe-missing'}">Gloves: ${ppe.gloves ? 'OK' : 'MISS'}</div>
                                <div class="ppe-item ${ppe.boots ? 'ppe-ok' : 'ppe-missing'}">Boots: ${ppe.boots ? 'OK' : 'MISS'}</div>
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
                    if (row.alert_type.includes('SAFE')) badgeClass = 'badge-safe';
                    return `<tr><td>${row.timestamp}</td><td><span class="badge ${badgeClass}">${row.alert_type}</span></td><td>${row.details}</td><td>${row.status}</td></tr>`;
                }).join('');
            } catch(e) { console.error('Update error:', e); }
        }
        setInterval(updateData, 1000);
        setInterval(updateAgent, 3000);
        updateData();
        updateAgent();

        async function updateAgent() {
            try {
                const a = await fetch('/api/agent').then(r => r.json());
                const panel = document.getElementById('agent-panel');
                if (a.length > 0) {
                    const latest = a[0];
                    const riskClass = 'agent-risk-' + (latest.risk_level || 'low').toLowerCase();
                    const modelBadge = latest.model_used 
                        ? `<span class="badge" style="background: #38bdf8; color: #0f172a; margin-left: 8px;">${latest.model_used}</span>` 
                        : '';
                    const responseTime = latest.response_time_ms 
                        ? `<div class="agent-tier">Response: ${latest.response_time_ms}ms</div>` 
                        : '';
                    const toolCalls = latest.tool_calls && latest.tool_calls.length > 0
                        ? `<div class="agent-tier" style="margin-top: 6px;">Tools: ${latest.tool_calls.map(t => t.name).join(', ')}</div>`
                        : '';
                    panel.innerHTML = `
                        <div style="display: flex; align-items: center; margin-bottom: 6px;">
                            <span class="${riskClass}" style="font-size: 16px;">${latest.risk_level || 'UNKNOWN'}</span>
                            ${modelBadge}
                        </div>
                        <div class="agent-reasoning">${latest.reasoning || (latest.decision_json && latest.decision_json.reasoning) || 'No reasoning available'}</div>
                        ${toolCalls}
                        <div class="agent-tier">Action Tier: ${latest.action_tier || (latest.decision_json && latest.decision_json.action_tier) || '-'}</div>
                        ${responseTime}
                        <div class="agent-tier" style="margin-top: 4px;">${latest.timestamp || ''}</div>
                    `;
                } else {
                    panel.innerHTML = '<div style="color: #64748b; text-align: center; padding: 20px;">No agent decisions yet</div>';
                }
            } catch(e) { console.error('Agent update error:', e); }
        }

        async function setAgentMode(mode) {
            try {
                const res = await fetch('/api/agent/mode', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({mode})
                });
                const data = await res.json();
                if (data.status === 'updated') {
                    updateModeUI(mode);
                }
            } catch(e) { console.error('Mode update error:', e); }
        }

        function updateModeUI(mode) {
            const badge = document.getElementById('agent-mode-badge');
            const btnOllama = document.getElementById('btn-ollama');
            const btnGroq = document.getElementById('btn-groq');
            if (mode === 'ollama') {
                badge.textContent = 'OLLAMA';
                badge.style.background = '#f59e0b';
                btnOllama.style.background = '#22c55e';
                btnOllama.style.color = '#0f172a';
                btnGroq.style.background = '#334155';
                btnGroq.style.color = '#e2e8f0';
            } else {
                badge.textContent = 'GROQ';
                badge.style.background = '#38bdf8';
                btnGroq.style.background = '#22c55e';
                btnGroq.style.color = '#0f172a';
                btnOllama.style.background = '#334155';
                btnOllama.style.color = '#e2e8f0';
            }
        }

        // Initialize button states on load
        updateModeUI('groq');
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

@app.route('/api/agent')
def api_agent():
    """Return latest agent decisions."""
    try:
        conn = sqlite3.connect('wasp.db', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM agent_decisions ORDER BY id DESC LIMIT 10")
        rows = c.fetchall()
        conn.close()
        decisions = []
        for r in rows:
            dj = json.loads(r["decision_json"]) if r["decision_json"] else {}
            decisions.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "context_json": r["context_json"],
                "decision_json": dj,
                "risk_level": r["risk_level"],
                "model_used": r["model_used"],
                "tool_calls": json.loads(r["tool_calls"]) if r["tool_calls"] else [],
                "response_time_ms": r["response_time_ms"],
                "reasoning": dj.get("reasoning", ""),
                "action_tier": dj.get("action_tier", None),
                "speak_bm": dj.get("speak_bm", ""),
                "speak_en": dj.get("speak_en", "")
            })
        return jsonify(decisions)
    except Exception as e:
        print(f"[API Agent Error] {e}")
        return jsonify([])

@app.route('/api/agent/mode', methods=['GET', 'POST'])
def agent_mode():
    """Get or set agent mode (ollama/groq)."""
    global AGENT_MODE
    if request.method == 'POST':
        data = request.get_json()
        mode = data.get("mode", "groq")
        if mode in ("ollama", "groq"):
            AGENT_MODE = mode
            print(f"[CONFIG] Agent mode switched to: {AGENT_MODE}")
            return jsonify({"mode": AGENT_MODE, "status": "updated"})
        return jsonify({"error": "Invalid mode. Use 'ollama' or 'groq'"}), 400
    else:
        return jsonify({"mode": AGENT_MODE})

def mqtt_recieve():
    global mqtt_client, mqtt_client_ready

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    if USERNAME:
        client.username_pw_set(USERNAME, PASSWORD)

    client.on_connect    = on_connect
    client.on_message    = on_message
    client.on_disconnect = on_disconnect

    # Store as global so mqtt_publish can reuse it
    with mqtt_client_lock:
        mqtt_client = client

    print(f"[MQTT] Connecting to {BROKER_HOST}:{BROKER_PORT} ...")
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
    os.makedirs("reports", exist_ok=True)

    init_db()
    init_tts()

    # Initialize WASP AI Agent
    print("[INIT] Initializing WASP AI Agent...")
    wasp_agent = WASPAgent()

    print(f"[INIT] Loading YOLOv8 model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        print(f"[INIT] Model loaded. Classes: {list(model.names.values())}")
        if CUDA_AVAILABLE:
            model.to('cuda')
            print("[INIT] YOLO running on GPU (CUDA)")
        else:
            print("[INIT] YOLO running on CPU")
    except Exception as e:
        print(f"[INIT] FATAL: Cannot load model: {e}")
        print("[INIT] Make sure the model file exists at the path above")
        exit(1)

    # Pre-warm Ollama model in background so first real call doesn't timeout
    def prewarm_ollama():
        try:
            requests.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": "ready", "stream": False}, timeout=120)
            print("[INIT] Ollama model pre-warmed successfully")
        except Exception as e:
            print(f"[INIT] Ollama pre-warm failed: {e}")
    threading.Thread(target=prewarm_ollama, daemon=True).start()

    print("[INIT] Starting CV thread...")
    threading.Thread(target=cv_thread, daemon=True).start()

    print("[INIT] Starting Sensor thread...")
    #threading.Thread(target=sensor_thread, daemon=True).start()

    print("[INIT] Starting Daily Report thread (fires at 18:00)...")
    threading.Thread(target=daily_report_loop, daemon=True).start()

    print("[INIT] Starting web server...")
    print("[INIT] Dashboard: http://localhost:5000")
    print("[INIT] Press Ctrl+C to stop")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
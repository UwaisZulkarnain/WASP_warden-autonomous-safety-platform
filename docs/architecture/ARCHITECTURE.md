# WASP System Architecture

Warden Autonomous Safety Platform — backend process, agent reasoning, and the hardware and legacy services it talks to.

The Flask backend runs three always-on threads against one shared agent: a **computer-vision thread** checks PPE compliance, a **sensor thread** watches the environment, and a **scheduled thread** compiles a daily report. All three hand their findings to the **agent engine**, which decides whether to log the event, speak a warning, or escalate — writing to SQLite, driving text-to-speech, and publishing alert commands back to the site hardware over MQTT.

```mermaid
flowchart TB
  subgraph BACKEND["WASP Backend — Flask process (wasp_backend.py)"]
    direction TB
    CV["CV Thread (YOLOv8)<br/>Person detection · PPE check · spatial matching"]
    SENSOR["Sensor Thread (Serial/MQTT)<br/>Temp · humidity · motion · IR/obstacle · sound · air quality"]
    REPORT["Daily Report Thread (18:00)<br/>JSON report · narrative report (Groq)"]
    AGENT{{"Agent Engine — WASPAgent<br/>Rule-based + Ollama/Groq + tool calling + ML anomaly detection"}}
    DB[("Database (SQLite)<br/>alerts · agent_decisions")]
    TTS["TTS Engine (pygame + gTTS)<br/>Pre-rendered MP3 (BM/EN)"]
    PUB["MQTT Publisher<br/>escalation commands"]
    DASH["Flask Web Dashboard — localhost:5000"]

    CV -->|PPE + person detections| AGENT
    SENSOR -->|sensor readings| AGENT
    REPORT -->|daily trigger 18:00| AGENT
    AGENT -->|writes alert + decision| DB
    AGENT -->|voice warning BM/EN| TTS
    AGENT -->|escalation command| PUB

    CV -.->|camera feed| DASH
    SENSOR -.->|env data| DASH
    AGENT -.->|reasoning| DASH
    DB -.->|alert history| DASH
  end

  ESP32["ESP32-S3 Camera (XIAO)<br/>Video stream · sensor array · buzzer/LED"]
  BROKER["MQTT Broker — port 1883"]
  NODERED["Node-RED Dashboard (legacy)<br/>Custom panels · flow-based logic"]

  ESP32 <-->|sensors + video in · alerts out| BACKEND
  BROKER <-->|publish / subscribe| BACKEND
  BACKEND -.->|legacy telemetry, read-only| NODERED
```

**Reading it:** a PPE or environmental anomaly travels through the ingestion threads into the agent engine, which decides to log, speak, or escalate; the dashboard reads live state from that same pipeline rather than sitting at the end of it.

A themed, hover-friendly version of this diagram (with a legend) is at [`wasp-system-architecture.html`](./wasp-system-architecture.html) — open it in a browser.

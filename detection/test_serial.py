"""Paen/Uwais: Use this to verify ESP32 sensor output"""
import serial
import json

ESP32_PORT = "COM3"  # CHANGE THIS - check Device Manager
BAUD_RATE = 115200

def main():
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
        print(f"[TEST] Connected to {ESP32_PORT} at {BAUD_RATE} baud")
        print("[TEST] Reading sensor data... Press Ctrl+C to stop")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[RAW] {line}")
                try:
                    data = json.loads(line)
                    print(f"[PARSED] Temp: {data.get('temperature')}C, Humidity: {data.get('humidity')}%, Motion: {data.get('motion')}")
                except json.JSONDecodeError:
                    print("[ERROR] Invalid JSON")
    except serial.SerialException as e:
        print(f"[ERROR] Cannot open {ESP32_PORT}: {e}")
        print("[HINT] Check Device Manager for correct COM port")
    except KeyboardInterrupt:
        print("\n[TEST] Stopped")

if __name__ == "__main__":
    main()

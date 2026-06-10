# WASP_warden-autonomous-safety-platform
Warden Autonomous Safety Platform — UTM FAI Showcase 2026

## Setting Up Camera in ESP32
Recommended use PlatformIO

In `platformio.ini` , paste the following:
```
[env:seeed_xiao_esp32s3]
platform = espressif32
board = seeed_xiao_esp32s3
framework = arduino
monitor_speed = 115200

lib_deps =
  espressif/esp32-camera @ ^2.0.0
```

Then, copy and upload code from `camera_setup_esp32.cpp` into `main.cpp`. Upload the code, then an IP address of ESP32 will be produced

## Setting Up cv_test_esp.py
Ensure in Python3 virtual environment all libraries are installed by running this code:
```
pip install opencv-python ultralytics paho-mqtt flask
```
Change info (`STREAM_URL`, `MQTT_BROKER`, etc) if needed. No change needed for `MQTT_Broker` if `cv_test_esp.py` and Node-Red running in same device

Run the `cv_test_esp.py` file. IP address should be produced

## Dashboard Setup
Import `flows.json` into Node-Red.
Ensure **Broker IP is same with IP address that runs cv_test_esp.py**
In `template` node, edit `img src = 'http://XXX.XXX.XXX.X/stream'` so that it matches with IP address produced from running `cv_test_esp.py`
Enter dashboard by adding `/dashboard` after `http://XXX.XXX.XXX.X` in browser

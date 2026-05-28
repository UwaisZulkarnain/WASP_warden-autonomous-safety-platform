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

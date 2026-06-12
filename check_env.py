"""Quick environment check before running WASP backend"""
import sys
import os

print("=" * 60)
print("WASP Environment Check")
print("=" * 60)

# Check Python version
print(f"\n[1] Python version: {sys.version}")
if sys.version_info < (3, 10):
    print("    WARNING: Python 3.10+ recommended")
else:
    print("    OK")

# Check required modules
modules = {
    "flask": "Flask",
    "cv2": "OpenCV",
    "ultralytics": "YOLOv8",
    "serial": "PySerial",
    "requests": "Requests",
    "gtts": "gTTS",
    "pygame": "Pygame",
    "numpy": "NumPy"
}

print("\n[2] Checking Python modules:")
all_ok = True
for mod, name in modules.items():
    try:
        __import__(mod)
        print(f"    [OK] {name}")
    except ImportError:
        print(f"    [MISSING] {name} — run: pip install -r requirements.txt")
        all_ok = False

# Check model file
print("\n[3] Checking model file:")
model_path = "computer_vision/best_construction_ppe.pt"
if os.path.exists(model_path):
    size = os.path.getsize(model_path) / (1024*1024)
    print(f"    [OK] {model_path} found ({size:.1f} MB)")
else:
    print(f"    [MISSING] {model_path} not found — copy Born's model to computer_vision/")
    all_ok = False

# Check warnings folder
print("\n[4] Checking TTS audio files:")
if os.path.exists("warnings") and os.listdir("warnings"):
    count = len(os.listdir("warnings"))
    print(f"    [OK] {count} audio files in warnings/")
else:
    print("    [MISSING] No audio files — run: python generate_tts.py")
    all_ok = False

# Check config
print("\n[5] Checking wasp_backend.py config:")
try:
    with open("wasp_backend.py", "r") as f:
        content = f.read()

    if "YOUR_APIKEY" in content:
        print("    [WARNING] CallMeBot API key not set — update wasp_backend.py")
    else:
        print("    [OK] API key appears configured")

    if "60123456789" in content:
        print("    [WARNING] Supervisor phone not set — update wasp_backend.py")
    else:
        print("    [OK] Phone number appears configured")

except FileNotFoundError:
    print("    [ERROR] wasp_backend.py not found in current folder")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("ALL CHECKS PASSED — Ready to run: python wasp_backend.py")
else:
    print("SOME CHECKS FAILED — Fix the issues above before running")
print("=" * 60)

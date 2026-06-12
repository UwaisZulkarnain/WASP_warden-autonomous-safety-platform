"""Generate pre-recorded BM warning audio files"""
from gtts import gTTS
import os

warnings = [
    "Perhatian! Helmet tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Vest tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Vest tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Suhu sangat tinggi. Sila berehat!",
]

os.makedirs("warnings", exist_ok=True)

for text in warnings:
    safe_name = text.replace(" ", "_").replace("!", "").replace(".", "").replace(":", "")[:50]
    filepath = f"warnings/{safe_name}.mp3"
    try:
        tts = gTTS(text=text, lang='ms')
        tts.save(filepath)
        print(f"[OK] Generated: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to generate {safe_name}: {e}")

print("\n[TTS] All warnings generated in warnings/ folder")
print("[TTS] Run this BEFORE the showcase. Internet required.")

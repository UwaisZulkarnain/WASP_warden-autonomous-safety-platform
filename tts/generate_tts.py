"""Generate pre-recorded BM warning audio files"""
from gtts import gTTS
import os

warnings = [
    "Perhatian! Helmet tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Harness tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Goggles tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Gloves tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Boots tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Harness tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Goggles tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Gloves tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Boots tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Harness dan Goggles tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Harness dan Goggles dan Gloves tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Helmet dan Harness dan Goggles dan Gloves dan Boots tidak dipakai. Sila pakai sekarang!",
    "Perhatian! Suhu sangat tinggi. Sila berehat!",
    "Perhatian! Suhu dan gas tinggi. Sila berehat dan periksa kawasan.",
    "Perhatian! Bacaan gas tinggi. Sila periksa pengudaraan.",
    "Perhatian! Kelembapan tinggi. Sila minum air dan berehat jika perlu.",
    "Ingatan! Sila pakai PPE yang lengkap.",
    "Ingatan! Sila lengkapkan PPE sebelum terus bekerja.",
]

os.makedirs("warnings", exist_ok=True)

for text in warnings:
    safe_name = text.replace(" ", "_").replace("!", "").replace(".", "").replace(":", "")
    filepath = f"warnings/{safe_name}.mp3"
    try:
        tts = gTTS(text=text, lang='ms')
        tts.save(filepath)
        print(f"[OK] Generated: {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to generate {safe_name}: {e}")

print("\n[TTS] All warnings generated in warnings/ folder")
print("[TTS] Run this BEFORE the showcase. Internet required.")

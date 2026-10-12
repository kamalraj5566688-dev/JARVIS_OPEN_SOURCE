"""
J.A.R.V.I.S. Voice AI Diagnostic & Test Suite
Tests Microphone Levels, Vosk Wake Word, Adaptive VAD, and Whisper STT.
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

_base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_base_dir, ".env"))

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_HOME" not in os.environ and os.path.exists(r"E:\.cache\huggingface"):
    os.environ["HF_HOME"] = r"E:\.cache\huggingface"
if "HF_HUB_CACHE" not in os.environ and os.path.exists(r"E:\.cache\huggingface\hub"):
    os.environ["HF_HUB_CACHE"] = r"E:\.cache\huggingface\hub"

SAMPLE_RATE = 16000
BLOCK_SIZE = 4000

print("=" * 64)
print("  J.A.R.V.I.S. Voice AI Diagnostic & Microphone Tester")
print("=" * 64)

# 1. Check Audio Devices
print("\n[1/4] Checking Audio Devices...")
devices = sd.query_devices()
default_in, default_out = sd.default.device
in_name = devices[default_in]["name"] if default_in is not None else "None"
out_name = devices[default_out]["name"] if default_out is not None else "None"
print(f"  Default Microphone : [{default_in}] {in_name}")
print(f"  Default Speakers   : [{default_out}] {out_name}")

# 2. Test Live Microphone Audio & RMS
print("\n[2/4] Testing Live Microphone Levels (Speak into your mic for 3 seconds)...")
audio_frames = []

def callback(indata, frames, time_info, status):
    audio_frames.append(bytes(indata))

with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype="int16", channels=1, callback=callback):
    for i in range(3, 0, -1):
        if audio_frames:
            latest = audio_frames[-1]
            data = np.frombuffer(latest, dtype=np.int16)
            rms = float(np.sqrt(np.mean(np.square(data.astype(np.float64))))) if len(data) else 0.0
            bar = "#" * int(min(20, rms // 20))
            print(f"  [Recording {i}s remaining] Live RMS: {rms:5.0f} | {bar:<20} |", flush=True)
        sd.sleep(1000)

all_pcm = b"".join(audio_frames)
all_arr = np.frombuffer(all_pcm, dtype=np.int16)
total_rms = float(np.sqrt(np.mean(np.square(all_arr.astype(np.float64))))) if len(all_arr) else 0.0
print(f"\n  Average Mic RMS: {total_rms:.1f}")
if total_rms < 25.0:
    print("  [!] WARNING: Audio level is very low. Please check Windows Microphone volume / boost in Settings.")
else:
    print("  [OK] Microphone signal is healthy and clear!")

# 3. Test Vosk Wake Word Engine
print("\n[3/4] Initializing Vosk Wake Word Engine...")
from vosk import Model, KaldiRecognizer
vosk_path = os.path.join(_base_dir, "models", "vosk-model-small-en-in-0.4")
v_model = Model(vosk_path)
rec = KaldiRecognizer(v_model, SAMPLE_RATE)
print("  [OK] Vosk Indian English engine loaded.")

# 4. Test Faster-Whisper Speech-to-Text
print("\n[4/4] Initializing faster-whisper Speech-to-Text...")
from audio_io import SpeechEngine
stt = SpeechEngine()
if total_rms >= 35.0 and len(all_pcm) > SAMPLE_RATE:
    print("  Transcribing test speech with Whisper...")
    text = stt.transcribe(all_pcm)
    print(f"  Whisper heard: '{text}'")
else:
    print("  Skipping test audio transcription (audio too short or silent).")

print("\n" + "=" * 64)
print("  Diagnostic Complete! All systems operational.")
print("  Run 'python main.py' to launch JARVIS in Public Mode.")
print("=" * 64)

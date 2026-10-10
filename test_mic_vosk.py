import json
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

print("1. Available Input Devices:")
devices = sd.query_devices()
for idx, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"  [{idx}] {d['name']} (Default: {idx == sd.default.device[0]})")

print("\n2. Loading Vosk Model...")
model = Model("models/vosk-model-small-en-in-0.4")
rec = KaldiRecognizer(model, 16000)

print("\n3. Testing live audio & mic levels...")
print("   Speak into your microphone now (say 'Hello Jaddu' or anything).")
print("   Press Ctrl+C to stop testing.\n")

def callback(indata, frames, time, status):
    if status:
        print(f"[Status]: {status}")
    raw_bytes = bytes(indata)
    arr = np.frombuffer(raw_bytes, dtype=np.int16)
    rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float64))))) if len(arr) else 0.0

    if rec.AcceptWaveform(raw_bytes):
        res = json.loads(rec.Result())
        text = res.get('text', '')
        if text:
            print(f"\n  >>> Recognized: '{text}' (RMS: {rms:.0f})")
    else:
        part = json.loads(rec.PartialResult())
        p_text = part.get("partial", "")
        if p_text:
            print(f"  [Hearing]: '{p_text}' (RMS: {rms:.0f})   ", end="\r", flush=True)

with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16', channels=1, callback=callback):
    try:
        while True:
            sd.sleep(500)
    except KeyboardInterrupt:
        pass

print("\n\nDone testing.")
"""
setup_models.py
===============
Utility script to automatically download and unpack the offline Vosk speech model
required for low-power wake-word detection.
"""

import os
import sys
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip"
MODEL_DIR = "models"
MODEL_NAME = "vosk-model-small-en-in-0.4"
TARGET_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
ZIP_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.zip")


def setup_vosk_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(TARGET_PATH):
        print(f"[+] Vosk model already exists at '{TARGET_PATH}'. No action required.")
        return

    print("=======================================================")
    print(" J.A.R.V.I.S. Offline Model Setup")
    print("=======================================================")
    print(f"[1/3] Downloading offline speech model from:\n      {MODEL_URL}...")

    def progress(blocks, block_size, total_size):
        downloaded = blocks * block_size
        pct = (downloaded / total_size) * 100 if total_size > 0 else 0
        sys.stdout.write(f"\r      Progress: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, ZIP_PATH, reporthook=progress)
        print("\n[2/3] Extracting archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

        print(f"[3/3] Done! Vosk model successfully installed to '{TARGET_PATH}'.")
    except Exception as e:
        print(f"\n[!] Error during model setup: {e}")
        print("[!] You can manually download the model from:")
        print(f"    {MODEL_URL}")
        print(f"    and extract it into the '{MODEL_DIR}/' directory.")


if __name__ == "__main__":
    setup_vosk_model()

import os
import numpy as np
import pyttsx3


class SpeechEngine:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        # Initialize Text-To-Speech engine with JARVIS voice profile
        self.tts = pyttsx3.init()
        try:
            # SAPI5 voice selection (select British voice if present, else male David)
            voices = self.tts.getProperty("voices")
            preferred_voice = None
            for v in voices:
                v_name = v.name.lower()
                if "british" in v_name or "george" in v_name or "hazel" in v_name:
                    preferred_voice = v.id
                    break
                elif "david" in v_name and not preferred_voice:
                    preferred_voice = v.id
            if preferred_voice:
                self.tts.setProperty("voice", preferred_voice)

            # JARVIS pacing: measured, crisp, calm
            self.tts.setProperty("rate", 175)
            self.tts.setProperty("volume", 0.95)
        except Exception as e:
            print(f"[Warning] SAPI5 configuration note: {e}")
        
        # Initialize STT engine (faster-whisper)
        self.whisper_model = None
        try:
            from faster_whisper import WhisperModel
            hf_hub = os.environ.get("HF_HUB_CACHE", "E:/.cache/huggingface/hub")
            self.whisper_model = WhisperModel(model_size, download_root=hf_hub, device=device, compute_type=compute_type)
        except Exception as e:
            print(f"[Warning] Could not initialize faster_whisper: {e}")

    def speak(self, text: str):
        """Converts text to speech and prints response."""
        print(f"[J.A.R.V.I.S.]: {text}")
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"[TTS Error]: {e}")

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribes PCM 16-bit mono audio bytes into text."""
        if not self.whisper_model:
            print("[SpeechEngine] Whisper model unavailable.")
            return ""
        
        try:
            # Convert 16-bit PCM audio bytes to float32 NumPy array normalized to [-1, 1]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(
                audio_np,
                beam_size=5,
                initial_prompt=(
                    "J.A.R.V.I.S. desktop assistant. Commands: launch VS Code, Chrome, Terminal, "
                    "system briefing, system status, battery, volume up, volume down, mute, screenshot, "
                    "minimize, lock, coding mode, break mode, remember that."
                )
            )
            text = " ".join([segment.text for segment in segments]).strip()
            return text
        except Exception as e:
            print(f"[Transcribe Error]: {e}")
            return ""

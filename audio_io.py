import os
import re
import numpy as np
import pyttsx3
from dotenv import load_dotenv

# Ensure environment variables are loaded
_base_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_base_dir, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_HOME" not in os.environ and os.path.exists(r"E:\.cache\huggingface"):
    os.environ["HF_HOME"] = r"E:\.cache\huggingface"
if "HF_HUB_CACHE" not in os.environ and os.path.exists(r"E:\.cache\huggingface\hub"):
    os.environ["HF_HUB_CACHE"] = r"E:\.cache\huggingface\hub"


class SpeechEngine:
    def __init__(self, model_size="base.en", device="cpu", compute_type="int8"):
        # Initialize Text-To-Speech engine with JARVIS voice profile
        self.tts = pyttsx3.init()
        try:
            # SAPI5 voice selection (prefer David or British voice)
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
            hf_hub = os.environ.get("HF_HUB_CACHE", r"E:\.cache\huggingface\hub")
            if not os.path.exists(hf_hub):
                hf_hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            try:
                self.whisper_model = WhisperModel(
                    model_size, download_root=hf_hub, device=device, compute_type=compute_type, local_files_only=True
                )
            except Exception:
                self.whisper_model = WhisperModel(
                    model_size, download_root=hf_hub, device=device, compute_type=compute_type
                )
        except Exception as e:
            print(f"[Warning] Could not initialize faster_whisper: {e}")

    def speak(self, text: str):
        """Converts text to speech and prints response."""
        print(f"[J.A.R.V.I.S.]: {text}", flush=True)
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            print(f"[TTS Error]: {e}", flush=True)

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribes PCM 16-bit mono audio bytes into text."""
        if not self.whisper_model:
            print("[SpeechEngine] Whisper model unavailable.", flush=True)
            return ""
        
        try:
            # Convert 16-bit PCM audio bytes to float32 NumPy array normalized to [-1, 1]
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.whisper_model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
                initial_prompt=(
                    "Jarvis AI assistant. Commands: launch VS Code, Chrome, Terminal, Notepad, Spotify, "
                    "what is the time, what is today's date, weather in London, play music on YouTube, "
                    "system briefing, system status, battery level, CPU, RAM, volume up, volume down, mute, "
                    "take a screenshot, minimize window, lock workstation, coding mode, break mode, remember that."
                )
            )
            raw_text = " ".join([segment.text for segment in segments]).strip()

            # Filter known Whisper phantom hallucination artifacts
            hallucinations = [
                "thank you for watching", "thanks for watching", "please subscribe", 
                "like and subscribe", "subtitles by", "[music]", "(music)", 
                "[applause]", "(bell chime)", "you"
            ]
            clean = raw_text.strip()
            for h in hallucinations:
                if clean.lower() == h:
                    return ""
                clean = re.sub(re.escape(h), "", clean, flags=re.IGNORECASE).strip()
            return clean
        except Exception as e:
            print(f"[Transcribe Error]: {e}", flush=True)
            return ""

import os
import sys
import subprocess

# Auto-delegate to local .venv if launched via global system python
if sys.prefix == sys.base_prefix:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_py) and os.path.normpath(sys.executable).lower() != os.path.normpath(venv_py).lower():
        sys.exit(subprocess.run([venv_py] + sys.argv).returncode)

# Load environment configuration (.env)
from dotenv import load_dotenv
_base_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_base_dir, ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path)

# Configure Hugging Face and Ollama environment paths
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_HOME" not in os.environ and os.path.exists(r"E:\.cache\huggingface"):
    os.environ["HF_HOME"] = r"E:\.cache\huggingface"
if "HF_HUB_CACHE" not in os.environ and os.path.exists(r"E:\.cache\huggingface\hub"):
    os.environ["HF_HUB_CACHE"] = r"E:\.cache\huggingface\hub"

import json
import queue
import re
import time
import difflib
from collections import deque

import numpy as np
import sounddevice as sd
import ollama
from vosk import Model as VoskModel, KaldiRecognizer

import msvcrt

from audio_io import SpeechEngine
from actions import DesktopController
from overlay import start_overlay
from jarvis_memory import JarvisMemory
from trace_logger import TraceLogger
from auto_trainer import auto_trainer
from jarvis_brain import jarvis_brain

# ----------------- Configuration & Audio Setup -----------------
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 250ms audio blocks
# Voice authentication is DISABLED for open public use (anyone can speak)
VOICE_AUTH_ENABLED = os.environ.get("JARVIS_REQUIRE_VOICE_AUTH", "0").strip() in ("1", "true", "True")

OLLAMA_HOST = os.environ.get("JARVIS_OLLAMA_HOST", os.environ.get("JADDU_OLLAMA_HOST", "http://localhost:11434"))
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL", "jarvis")
OLLAMA_MODEL_PREFERENCE = [
    "jarvis", "qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b",
    "qwen2.5", "llama3.2", "llama3.1", "mistral", "gemma"
]

print("Loading wake word engine (Vosk Indian English)...")
vosk_model_path = os.path.join(_base_dir, "models", "vosk-model-small-en-in-0.4")
if not os.path.exists(vosk_model_path):
    vosk_model_path = "models/vosk-model-small-en-in-0.4"
vosk_model = VoskModel(vosk_model_path)
rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

print("Initializing Speech-to-Text engine (faster-whisper)...")
speech_engine = SpeechEngine(model_size="base.en", device="cpu", compute_type="int8")

# Optional voice verifier (only active if user explicitly configures JARVIS_REQUIRE_VOICE_AUTH=1)
speaker = None
if VOICE_AUTH_ENABLED:
    try:
        from speaker_verifier import SpeakerVerifier
        speaker = SpeakerVerifier(profile_path="voice_profile.npy")
        print("[+] Voice authentication enabled.")
    except Exception as e:
        print(f"[!] Voice verifier unavailable: {e}. Defaulting to open mode.")
        VOICE_AUTH_ENABLED = False
else:
    print("[+] Public Mode: Voice authentication removed (Ready for everyone).")

print("Initializing OpenJarvis Memory & Trace Logger...")
memory = JarvisMemory()
trace_logger = TraceLogger()

audio_queue = queue.Queue()

# ----------------- Ollama (local LLM) setup -----------------
os.environ["OLLAMA_HOST"] = OLLAMA_HOST


def ensure_ollama_running():
    """Checks if Ollama is running; if not, automatically launches 'ollama serve' in background."""
    try:
        ollama.list()
        return True
    except Exception:
        pass

    print("[Ollama] Daemon not running, launching in background...", flush=True)
    try:
        ollama_bin = "ollama"
        prog_path = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
        if os.path.exists(prog_path):
            ollama_bin = prog_path
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        subprocess.Popen(
            [ollama_bin, "serve"],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(8):
            time.sleep(0.5)
            try:
                ollama.list()
                print("[Ollama] Daemon started successfully.", flush=True)
                return True
            except Exception:
                pass
    except Exception as e:
        print(f"[Ollama] Could not auto-start daemon: {e}", flush=True)
    return False


def poll_ollama_models():
    """Returns the list of models currently available in local Ollama."""
    ensure_ollama_running()
    try:
        resp = ollama.list()
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        seen = set()
        models = []
        for m in data.get("models", []):
            name = m.get("model") or m.get("name") or ""
            if not name:
                continue
            base = name.split(":")[0] if ":" in name else name
            for variant in (base, name):
                if variant not in seen:
                    seen.add(variant)
                    models.append(variant)
        return models
    except Exception as e:
        print(f"[Ollama] Daemon not reachable at {OLLAMA_HOST}: {e}")
        return []


_available_models = poll_ollama_models()
if _available_models:
    if OLLAMA_MODEL in _available_models:
        ACTIVE_OLLAMA_MODEL = OLLAMA_MODEL
    else:
        ACTIVE_OLLAMA_MODEL = next(
            (m for m in OLLAMA_MODEL_PREFERENCE if m in _available_models),
            _available_models[0],
        )
    print(f"[Ollama] Using model: {ACTIVE_OLLAMA_MODEL}")
else:
    ACTIVE_OLLAMA_MODEL = OLLAMA_MODEL
    print(f"[Ollama] Using configured model: {ACTIVE_OLLAMA_MODEL} (daemon offline)")


def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    audio_queue.put(bytes(indata))


def calculate_rms(audio_bytes: bytes) -> float:
    """Calculates RMS energy to detect speech vs silence."""
    data = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data.astype(np.float64)))))


# ----------------- OpenJarvis Tool Definitions -----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Opens desktop applications such as VS Code, Terminal, Chrome, Notepad, Calculator, Spotify, or Explorer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_target": {
                        "type": "string",
                        "description": "Target application name (e.g. 'vscode', 'chrome', 'terminal', 'notepad', 'calculator', 'spotify')"
                    }
                },
                "required": ["app_target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_site",
            "description": "Opens a website or web destination in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Domain name or site name (e.g. 'youtube', 'github', 'linkedin')"
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches Google or the web for general queries, information, or topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'weather today', 'latest tech news', 'python docs')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Controls audio volume, mute toggle, or media play/pause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "mute", "play/pause", "next", "prev"],
                        "description": "Media action to perform"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "window_management",
            "description": "Manages windows: close current window, minimize, switch window, show desktop, or lock machine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["close", "minimize", "maximize", "switch", "desktop", "lock"],
                        "description": "Window management operation"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_status",
            "description": "Gets real-time telemetry: CPU utilization, RAM, Disk, and Battery state.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_briefing",
            "description": "Provides a comprehensive JARVIS morning/system briefing including time, battery, and hardware telemetry.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Saves a persistent fact or preference to long-term memory across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_key": {
                        "type": "string",
                        "description": "Subject or title of the fact to remember (e.g. 'user_project', 'favorite_browser')"
                    },
                    "fact_value": {
                        "type": "string",
                        "description": "Details or value of the fact to remember"
                    }
                },
                "required": ["fact_key", "fact_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Captures a screenshot of the display and saves it to the desktop.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Runs a safe terminal/shell command (e.g. git status, dir, ipconfig).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command string to execute in shell"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_macro",
            "description": "Triggers multi-step workflow presets like 'coding mode', 'break mode', or 'wrap up work'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode_name": {
                        "type": "string",
                        "description": "Preset name or intent (e.g., 'coding', 'break', 'wrap up')"
                    }
                },
                "required": ["mode_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "learning_status",
            "description": "Reports the autonomous self-training and continual learning progress, cycles, and traces count.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_self_training",
            "description": "Triggers an immediate autonomous background fine-tuning cycle on the local GPU.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


# ----------------- Execution & Routing -----------------
def _parse_tool_call(content: str):
    """Parses tool calls emitted in content (<tool_call> or bare JSON)."""
    if not content:
        return None

    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    raw_json = match.group(1).strip() if match else None

    if not raw_json:
        candidate = content.strip().strip("`")
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("name"):
                raw_json = candidate
        except json.JSONDecodeError:
            pass

    if not raw_json:
        return None

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    func_name = data.get("name") if isinstance(data, dict) else None
    if not func_name:
        return None

    args = data.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return func_name, args


conversation_history = deque(maxlen=6)


def execute_intent(command: str) -> str:
    """Routes a command through the JarvisBrain cognitive and tool engine with error recovery."""
    jarvis_brain.model_name = ACTIVE_OLLAMA_MODEL
    jarvis_brain.ollama_host = OLLAMA_HOST
    try:
        return jarvis_brain.process_query(command)
    except Exception as e:
        print(f"[Main Warning] Error processing query via brain: {e}. Executing heuristic fallback...", flush=True)
        fallback = jarvis_brain.fast_heuristic_fallback(command)
        if fallback:
            return fallback
        return f"I encountered an error executing that request, sir: {str(e)}"


def _fallback_intent(command: str) -> str:
    """Matches desktop commands directly using JarvisBrain's fast heuristic reflex."""
    return jarvis_brain.fast_heuristic_fallback(command) or ""


def _drain_queue():
    """Removes all buffered audio frames (used before/after TTS to prevent echo)."""
    time.sleep(0.12)
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def capture_command(existing_audio_prefix: bytes = None) -> str:
    """
    Captures user spoken command with dynamic adaptive VAD.
    Adapts automatically to microphone sensitivity and ambient noise.
    """
    _drain_queue()

    # Dynamic ambient noise estimation
    ambient_samples = []
    for _ in range(3):
        try:
            sample = audio_queue.get(timeout=0.4)
            ambient_samples.append(calculate_rms(sample))
        except queue.Empty:
            break
    
    ambient_rms = float(np.mean(ambient_samples)) if ambient_samples else 35.0
    # Adaptive threshold: ambient floor + 10, clamped between 42.0 and 75.0
    speech_threshold_rms = max(42.0, min(75.0, ambient_rms * 1.35 + 8.0))

    preroll_buffer = deque(maxlen=4)  # ~1.0s buffer to preserve initial consonant
    recorded_frames = []
    if existing_audio_prefix:
        recorded_frames.append(existing_audio_prefix)

    silence_frames = 0
    speech_started = len(recorded_frames) > 0
    max_wait_frames = 32  # ~8.0s max wait before user starts speaking
    max_speech_frames = 56  # ~14s max command duration

    print(f"\n[VAD] Listening for command... (Ambient: {ambient_rms:.0f}, Speech threshold: {speech_threshold_rms:.0f})", flush=True)

    while True:
        try:
            data = audio_queue.get(timeout=1.0)
        except queue.Empty:
            break

        rms = calculate_rms(data)

        if not speech_started:
            preroll_buffer.append(data)
            # Live volume indicator
            bar_len = int(min(20, rms // 10))
            bar = "=" * bar_len + " " * (20 - bar_len)
            print(f"  [Mic Level]: [{bar}] RMS: {rms:4.0f} / Threshold: {speech_threshold_rms:.0f}  \r", end="", flush=True)

            if rms >= speech_threshold_rms:
                speech_started = True
                print(f"\n[VAD] Speech detected (RMS: {rms:.0f}), recording...", flush=True)
                recorded_frames.extend(preroll_buffer)
            else:
                max_wait_frames -= 1
                if max_wait_frames <= 0:
                    print(f"\n[VAD] Timed out waiting for speech.", flush=True)
                    break
        else:
            recorded_frames.append(data)
            bar_len = int(min(20, rms // 10))
            bar = "#" * bar_len + " " * (20 - bar_len)
            print(f"  [Recording]: [{bar}] RMS: {rms:4.0f}  \r", end="", flush=True)

            if rms < speech_threshold_rms:
                silence_frames += 1
            else:
                silence_frames = 0

            # 6 frames of silence (~1.5s) signals end of speech (natural human pause)
            if silence_frames >= 6:
                print(f"\n[VAD] End of speech detected ({len(recorded_frames) * 0.25:.1f}s captured).", flush=True)
                break
            if len(recorded_frames) >= max_speech_frames:
                print("\n[VAD] Max command duration reached.", flush=True)
                break

    if not recorded_frames:
        return ""

    full_audio = b"".join(recorded_frames)
    print("[STT] Transcribing audio with Whisper...", flush=True)
    text = speech_engine.transcribe(full_audio) or ""
    return text.strip()


def handle_wake(wake_audio, ui_pill, pre_extracted_command: str = ""):
    """
    Wake interaction flow: Open public access for everyone!
    Instantly transitions to LISTENING without biometric rejections.
    """
    ui_pill.set_state("LISTENING")

    # If the user spoke the entire command in one breath (e.g. "Jarvis open Chrome")
    if pre_extracted_command and len(pre_extracted_command.strip().split()) >= 2:
        command = pre_extracted_command.strip()
        print(f"[Direct Command]: '{command}'", flush=True)
    else:
        # User called wake word alone -> acknowledge and listen
        user_name = memory.get_user_name() or "Sir"
        speech_engine.speak(f"At your service, {user_name}.")
        _drain_queue()

        command = capture_command()
        print(f"[User]: '{command}'", flush=True)

    if not command or len(command.strip()) == 0:
        response_text = "I didn't catch that, sir. Standing by."
    else:
        ui_pill.set_state("THINKING")
        response_text = execute_intent(command)

    ui_pill.set_state("SPEAKING")
    speech_engine.speak(response_text)
    ui_pill.set_state("IDLE")
    _drain_queue()

    # Autonomous self-training check after speech turn
    auto_trainer.on_interaction(voice_notifier=speech_engine)


def main():
    ui_pill = start_overlay()
    ui_pill.set_state("IDLE")

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback
    )

    print("\n=======================================================", flush=True)
    print(" J.A.R.V.I.S. is Online! (OpenJarvis Architecture)", flush=True)
    print(" [Public Mode] Ready for everyone - Voice auth removed.", flush=True)
    print(" Call 'Jarvis' or 'Hey Jarvis' to activate.", flush=True)
    print(" OR Press [ENTER] / [SPACE] in this terminal anytime!", flush=True)
    print(" Press Ctrl+C in terminal to exit.", flush=True)
    print("=======================================================\n", flush=True)

    # Comprehensive phonetic patterns for "Jarvis" / "Hey Jarvis" / "Jaddu"
    # Kaldi Indian English acoustic model frequently mishears "Jarvis" in many ways;
    # all observed misdecodes are listed here to maximise wake detection.
    wake_phonetic_patterns = [
        # Standard pronunciations
        "jarvis", "hey jarvis", "hello jarvis", "hi jarvis", "ok jarvis", "okay jarvis",
        "jar vis", "jar visual", "jar base", "jarish", "jarves", "jarvez", "jarv",
        # Common Kaldi Indian English acoustic misdecodes of "Jarvis"
        "service", "services", "surface", "travis", "harvest", "charvis", "jervis",
        "star wars", "jar was", "jar is", "javis", "charles", "justice", "drivers",
        "jersey", "jared", "damage", "device", "diverse", "davis", "chavez", "chaves",
        "javed", "java", "jerry", "charlie", "address",
        "hey service", "hello service", "hey travis", "hey jersey", "hello jersey",
        # Additional observed misdecodes from vosk-model-small-en-in-0.4
        # 'J' onset often decoded as: 'after', 'officer', 'arthur', 'author'
        # 'arvis' suffix decoded as: 'opening', 'office', 'often'
        "after", "officer", "arthur", "author", "office",
        "after opening", "after office", "after often",
        "hey after", "hello after", "hey officer", "hello officer",
        "harbour", "harvest", "carve", "garvis", "garb", "karvis",
        # Backward-compatible Jaddu variants
        "jaddu", "jadu", "judo", "jadoo", "hey jaddu", "hello jaddu", "hi jaddu",
        "hello do", "hello due", "hello dew", "hello judo", "hello dude"
    ]

    # Action verbs that — when found as the FIRST real word in a final Vosk result —
    # indicate this was a command even if the wake word was completely swallowed.
    _action_prefixes = {
        "open", "launch", "start", "play", "show", "check", "get",
        "take", "search", "find", "run", "close", "minimize", "maximize",
        "lock", "mute", "unmute", "increase", "decrease", "volume",
        "remember", "what", "tell", "give", "set", "switch"
    }

    def extract_wake_and_command(text: str):
        """
        Detects if text contains a wake word, and extracts any trailing command.
        Returns (is_wake: bool, trailing_command: str)
        """
        t = text.lower().strip()
        if not t:
            return False, ""

        # Direct pattern check
        for pattern in wake_phonetic_patterns:
            if pattern in t:
                # Extract command spoken after the wake word
                idx = t.find(pattern)
                after = t[idx + len(pattern):].strip()
                return True, after

        words = t.split()
        for idx, w in enumerate(words):
            # Prefix matches
            if w.startswith("jar") or w.startswith("jer") or w.startswith("jadd") or w.startswith("jav"):
                after = " ".join(words[idx + 1:]).strip()
                return True, after

            # Specific common Kaldi one-word decodings of "Jarvis"
            if w in ["jarvis", "jersey", "jared", "damage", "jervis", "javis",
                     "charvis", "service", "travis", "harvest", "jaddu", "jadu",
                     "judo", "drivers", "chavez", "davis", "after", "officer",
                     "arthur", "author", "harbour", "garvis", "karvis"]:
                after = " ".join(words[idx + 1:]).strip()
                return True, after

            # Lowered fuzzy threshold (0.62) to catch more acoustic variants
            sim_jarvis = difflib.SequenceMatcher(None, w, "jarvis").ratio()
            sim_jaddu  = difflib.SequenceMatcher(None, w, "jaddu").ratio()
            if sim_jarvis >= 0.62 or sim_jaddu >= 0.62:
                after = " ".join(words[idx + 1:]).strip()
                return True, after

        # Short greetings (1-3 words): "hello jarvis", "hey jarvis", etc.
        if 1 <= len(words) <= 3 and words[0] in ["hello", "hey", "hi", "halo", "ok", "okay"]:
            if len(words) >= 2:
                w2 = words[1]
                if (difflib.SequenceMatcher(None, w2, "jarvis").ratio() >= 0.60 or
                        w2.startswith("j") or
                        w2 in ["service", "travis", "jersey", "jared",
                               "after", "officer", "arthur", "author"]):
                    after = " ".join(words[2:]).strip()
                    return True, after

        # LAST-RESORT: If the FINAL Vosk result (≤10 words) begins with a clear
        # action verb and no wake word was detected above, treat the whole
        # utterance as the command. This catches cases where the acoustic model
        # completely swallowed the wake-word prefix.
        if words and words[0] in _action_prefixes and len(words) <= 10:
            return True, t

        return False, ""

    wake_ring = deque(maxlen=int(2.0 * SAMPLE_RATE / BLOCK_SIZE))

    with stream:
        print("[Vosk] Listening for wake word ('Jarvis' or press ENTER)...", flush=True)
        while True:
            # Instant Keyboard trigger: Press ENTER or Space in terminal anytime
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in [b'\r', b'\n', b' ']:
                    print("\n[!] Keyboard trigger activated! Listening for command...", flush=True)
                    rec.Reset()
                    _drain_queue()
                    handle_wake(None, ui_pill, pre_extracted_command="")
                    wake_ring.clear()
                    rec.Reset()
                    print("\n[Vosk] Listening for 'Jarvis' (or press ENTER)...", flush=True)
                    continue

            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            wake_ring.append(data)

            heard_text = ""
            is_final = False
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                heard_text = res.get("text", "").lower().strip()
                is_final = True
                if heard_text:
                    print(f"\n[Vosk heard]: '{heard_text}'", flush=True)
            else:
                part = json.loads(rec.PartialResult())
                heard_text = part.get("partial", "").lower().strip()
                if heard_text:
                    print(f"  [Hearing]: {heard_text}   \r", end="", flush=True)

            if heard_text:
                triggered, command_tail = extract_wake_and_command(heard_text)
                if triggered:
                    print(f"\n[!] Wake word triggered: '{heard_text}'", flush=True)
                    rec.Reset()

                    wake_audio = b"".join(wake_ring)
                    _drain_queue()

                    handle_wake(wake_audio, ui_pill, pre_extracted_command=command_tail)

                    wake_ring.clear()
                    rec.Reset()
                    print("\n[Vosk] Listening for 'Jarvis' (or press ENTER)...", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJ.A.R.V.I.S. stopped.")

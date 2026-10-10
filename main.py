import os
import json
import queue
import re
import time
from collections import deque

import numpy as np
import sounddevice as sd
import ollama
from vosk import Model as VoskModel, KaldiRecognizer

import msvcrt

from audio_io import SpeechEngine
from actions import DesktopController
from overlay import start_overlay
from speaker_verifier import SpeakerVerifier
from jarvis_memory import JarvisMemory
from trace_logger import TraceLogger
from auto_trainer import auto_trainer

# ----------------- Configuration & Audio Setup -----------------
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000  # 250ms audio blocks
VOICE_PROFILE_PATH = "voice_profile.npy"
VERIFY_THRESHOLD = float(os.environ.get("JARVIS_VERIFY_THRESHOLD", "0.62"))
OLLAMA_HOST = os.environ.get("JARVIS_OLLAMA_HOST", os.environ.get("JADDU_OLLAMA_HOST", "http://localhost:11434"))
OLLAMA_MODEL = os.environ.get("JARVIS_OLLAMA_MODEL", "jarvis")
OLLAMA_MODEL_PREFERENCE = [
    "jarvis", "qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "llama3.2:1b",
    "qwen2.5", "llama3.2", "llama3.1", "mistral", "gemma"
]

print("Loading wake word engine (Vosk)...")
vosk_model = VoskModel("models/vosk-model-small-en-in-0.4")
rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

print("Initializing Speech-to-Text engine (Whisper)...")
speech_engine = SpeechEngine(model_size="base.en", device="cpu", compute_type="int8")

print("Initializing Speaker Verifier (voice biometrics)...")
speaker = SpeakerVerifier(profile_path=VOICE_PROFILE_PATH, threshold=VERIFY_THRESHOLD)
speaker.threshold = VERIFY_THRESHOLD
if not speaker.profile_exists:
    print(f"[!] No voice profile found at '{VOICE_PROFILE_PATH}'.")
    print(f"[!] Run `python enroll_voice.py` to enroll your voice.")
else:
    print(f"[+] Voice profile loaded "
          f"(samples={speaker.enrollment_metadata.get('n_samples')}, "
          f"created={speaker.enrollment_metadata.get('created_at')})")

print("Initializing OpenJarvis Memory & Trace Logger...")
memory = JarvisMemory()
trace_logger = TraceLogger()

audio_queue = queue.Queue()

# ----------------- Ollama (local LLM) setup -----------------
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
ollama.host = OLLAMA_HOST


def ensure_ollama_running():
    """Checks if Ollama is running; if not, automatically launches 'ollama serve' in background."""
    try:
        ollama.list()
        return True
    except Exception:
        pass

    print("[Ollama] Daemon not running, launching in background...", flush=True)
    try:
        import subprocess
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
    """Returns the list of models currently available in the local Ollama."""
    ensure_ollama_running()
    try:
        resp = ollama.list()
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        models = []
        for m in data.get("models", []):
            name = m.get("model") or m.get("name") or ""
            if name:
                models.append(name.split(":")[0] if ":" in name and name != "jarvis" else name)
                models.append(name)
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
        print(f"[Audio Status]: {status}", flush=True)
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
    """Routes a command through the local LLM with persistent memory, tool execution, and trace logging."""
    start_time = time.time()
    
    # 1. Inject OpenJarvis Persistent Memory
    memory_context = memory.build_system_context()

    system_prompt = (
        "You are J.A.R.V.I.S., the user's personal, ultra-competent desktop AI assistant inspired by Tony Stark's system.\n"
        "PERSONALITY: Cultured, loyal, articulate, calm, and subtly witty. Address the user respectfully as 'Sir' (or their designated name).\n"
        "BREVITY: You are a spoken voice assistant. Keep verbal responses strictly to 1-2 crisp, elegant sentences.\n"
        "TOOLS: Always call the matching tool when the user requests desktop actions (apps, media, websites, windows, system telemetry, screenshots, macros, memory storage, or shell commands).\n"
        f"{memory_context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(list(conversation_history))
    messages.append({"role": "user", "content": command})

    tool_used = None
    tool_args = None
    tool_output = None
    success = True

    try:
        response = ollama.chat(
            model=ACTIVE_OLLAMA_MODEL,
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0.2, "num_ctx": 4096, "num_predict": 100},
        )

        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        message = data.get("message", {})

        # Tool detection
        tool_call_info = None
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            tc = tool_calls[0]
            func_name = (tc.get("function") or {}).get("name")
            raw_args = (tc.get("function") or {}).get("arguments") or {}
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            if func_name:
                tool_call_info = (func_name, raw_args)

        if tool_call_info is None:
            tool_call_info = _parse_tool_call(message.get("content", ""))

        if tool_call_info:
            func_name, args = tool_call_info
            tool_used = func_name
            tool_args = args
            print(f"[Tool Triggered]: {func_name}({args})", flush=True)

            func = getattr(DesktopController, func_name, None)
            if func:
                result = func(**args)
                tool_output = result
                conversation_history.append({"role": "user", "content": command})
                conversation_history.append({"role": "assistant", "content": result})
                
                # Save to persistent memory & trace logger
                memory.record_turn(command, result, tool_used)
                latency = (time.time() - start_time) * 1000
                trace_logger.log_interaction(
                    query=command,
                    response=result,
                    tool_called=tool_used,
                    tool_args=tool_args,
                    tool_output=tool_output,
                    latency_ms=latency,
                    model_name=ACTIVE_OLLAMA_MODEL,
                    success=True
                )
                return result
            return "Command execution method could not be found, sir."

        content = (message.get("content") or "").strip()
        final_reply = content or "I am listening, sir, but did not catch your intent."
        conversation_history.append({"role": "user", "content": command})
        conversation_history.append({"role": "assistant", "content": final_reply})
        
        # Save to persistent memory & trace logger
        memory.record_turn(command, final_reply)
        latency = (time.time() - start_time) * 1000
        trace_logger.log_interaction(
            query=command,
            response=final_reply,
            latency_ms=latency,
            model_name=ACTIVE_OLLAMA_MODEL,
            success=True
        )
        return final_reply

    except Exception as e:
        print(f"[Ollama] LLM call unavailable ({e}), trying desktop fallback...", flush=True)
        fallback = _fallback_intent(command)
        if fallback:
            conversation_history.append({"role": "user", "content": command})
            conversation_history.append({"role": "assistant", "content": fallback})
            memory.record_turn(command, fallback)
            latency = (time.time() - start_time) * 1000
            trace_logger.log_interaction(
                query=command,
                response=fallback,
                latency_ms=latency,
                model_name="desktop_fallback",
                success=True
            )
            return fallback
        return "The local intelligence engine is currently offline, sir. Please start Ollama."


def _fallback_intent(command: str) -> str:
    """Matches common desktop commands directly if Ollama is unreachable."""
    cmd = command.lower().strip()
    if any(k in cmd for k in ["learning status", "training status", "how much trained", "auto train"]):
        return DesktopController.learning_status()
    if any(k in cmd for k in ["train yourself", "start training", "trigger training", "self train"]):
        return DesktopController.trigger_self_training()
    if any(k in cmd for k in ["briefing", "morning briefing", "system report"]):
        return DesktopController.system_briefing()
    if any(k in cmd for k in ["status", "system", "battery", "cpu", "ram"]):
        return DesktopController.system_status()
    if any(k in cmd for k in ["screenshot", "screen shot"]):
        return DesktopController.take_screenshot()
    if any(k in cmd for k in ["vs code", "vscode", "code"]):
        return DesktopController.launch_app("vscode")
    if any(k in cmd for k in ["chrome", "google chrome", "browser"]):
        return DesktopController.launch_app("chrome")
    if any(k in cmd for k in ["terminal", "powershell", "cmd"]):
        return DesktopController.launch_app("terminal")
    if any(k in cmd for k in ["notepad"]):
        return DesktopController.launch_app("notepad")
    if any(k in cmd for k in ["spotify"]):
        return DesktopController.launch_app("spotify")
    if any(k in cmd for k in ["mute", "unmute"]):
        return DesktopController.media_control("mute")
    if any(k in cmd for k in ["volume up", "increase volume", "louder"]):
        return DesktopController.media_control("up")
    if any(k in cmd for k in ["volume down", "lower volume", "quieter"]):
        return DesktopController.media_control("down")
    if any(k in cmd for k in ["minimize", "hide windows"]):
        return DesktopController.window_management("minimize")
    if any(k in cmd for k in ["desktop", "show desktop"]):
        return DesktopController.window_management("desktop")
    if any(k in cmd for k in ["lock", "lock pc", "lock workstation"]):
        return DesktopController.window_management("lock")
    if any(k in cmd for k in ["coding mode", "dev mode"]):
        return DesktopController.run_macro("coding")
    if any(k in cmd for k in ["break mode", "chill mode"]):
        return DesktopController.run_macro("chill")
    if "search" in cmd or "google" in cmd:
        q = cmd.replace("search", "").replace("google", "").replace("for", "").strip()
        if q:
            return DesktopController.search_web(q)
    return ""


def _drain_queue():
    """Removes all buffered audio frames (used before/after TTS)."""
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def capture_command() -> str:
    """Captures user spoken command via VAD state machine."""
    _drain_queue()

    speech_threshold_rms = 65.0  # sensitive enough for standard laptop/desktop mic (~30-50 ambient)
    preroll_buffer = deque(maxlen=4)  # ~1.0s buffer to preserve initial consonant
    recorded_frames = []
    silence_frames = 0
    speech_started = False
    max_wait_frames = 28  # ~7.0s max wait before user starts speaking
    max_speech_frames = 48  # ~12s max speech length

    print("\n[VAD] Listening for command (speak now)...", flush=True)

    while True:
        data = audio_queue.get()
        rms = calculate_rms(data)

        if not speech_started:
            preroll_buffer.append(data)
            if rms >= speech_threshold_rms:
                speech_started = True
                print(f"[VAD] Speech detected (RMS: {rms:.0f}), recording...", flush=True)
                recorded_frames.extend(preroll_buffer)
            else:
                max_wait_frames -= 1
                if max_wait_frames <= 0:
                    print(f"[VAD] Timed out waiting for speech (ambient RMS: {rms:.0f}).", flush=True)
                    break
        else:
            recorded_frames.append(data)
            if rms < speech_threshold_rms:
                silence_frames += 1
            else:
                silence_frames = 0

            if silence_frames >= 4:
                print(f"[VAD] End of speech detected ({len(recorded_frames) * 0.25:.1f}s captured).", flush=True)
                break
            if len(recorded_frames) >= max_speech_frames:
                print("[VAD] Max command duration reached.", flush=True)
                break

    if not recorded_frames:
        return ""

    full_audio = b"".join(recorded_frames)
    print("[STT] Transcribing audio with Whisper...", flush=True)
    text = speech_engine.transcribe(full_audio) or ""
    return text.strip()


def handle_wake(wake_audio, ui_pill):
    """Full wake interaction flow with voice biometrics, persistent memory, and Jarvis execution."""
    ui_pill.set_state("VERIFYING")

    # If wake triggered by keyboard or no profile exists, bypass verification
    if wake_audio is None or not speaker.profile_exists:
        if wake_audio is not None and not speaker.profile_exists:
            print("\n[Verify] Note: Running in OPEN mode (no voice profile enrolled).", flush=True)
        is_owner = True
    else:
        is_owner, similarity = speaker.verify(wake_audio)
        threshold = speaker.threshold
        print(f"\n[Verify] Similarity: {similarity:.3f} / threshold {threshold:.2f} -> "
              f"{'ACCEPTED' if is_owner else 'REJECTED'}", flush=True)

        if not is_owner:
            ui_pill.set_state("REJECTED")
            speech_engine.speak(
                "Voice authentication failed, sir. Access is restricted to the enrolled owner."
            )
            ui_pill.set_state("IDLE")
            _drain_queue()
            return

    # Authorized owner -> prompt and capture command
    ui_pill.set_state("LISTENING")
    user_name = memory.get_user_name()
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
    print(" Call 'Jarvis' or 'Hey Jarvis' to activate (or press ENTER in terminal).", flush=True)
    print(" Owner voice verification is active on your profile.", flush=True)
    print(" Press Ctrl+C in terminal to exit.", flush=True)
    print("=======================================================\n", flush=True)

    # Phonetic patterns recognized by Kaldi for "Jarvis" / "Hey Jarvis" + backward compatible "Jaddu"
    wake_phonetic_patterns = [
        "jarvis", "hey jarvis", "hello jarvis", "hi jarvis", "ok jarvis", "okay jarvis",
        "jar vis", "service", "travis", "harvest", "charvis", "jervis", "star wars",
        "jar was", "jar is", "javis", "charles", "justice", "drivers",
        # Observed Kaldi phonetic decodings for "Jarvis" with Indian accent
        "jersey", "jared", "damage", "hey jersey", "hello jersey", "hey jared", "hello jared",
        "hey damage", "hey service", "hello service",
        # Backward compatible patterns
        "hello do", "hello due", "hello dew", "hello judo", "hello dude", "hello jaddu",
        "hello jadu", "hey do", "hey judo", "hey jaddu", "hey jadu", "hi jaddu",
        "judo", "jaddu", "jadu", "jadoo"
    ]

    def is_wake_word(text: str) -> bool:
        t = text.lower().strip()
        if not t:
            return False
        # Direct pattern match
        if any(p in t for p in wake_phonetic_patterns):
            return True
        words = t.split()
        # Word-level phonetic matching
        for w in words:
            if w in [
                "jarvis", "jersey", "jared", "damage", "jervis", "javis",
                "charvis", "service", "travis", "harvest", "jaddu", "jadu", "judo", "drivers"
            ]:
                return True
            if w.startswith("jar") or w.startswith("jer") or w.startswith("jadd"):
                return True
        # Matches any short greeting (1-3 words) starting with hello/hey/hi/ok/okay/jarvis
        if 1 <= len(words) <= 3 and words[0] in [
            "hello", "hey", "hi", "halo", "ok", "okay", "jarvis", "jersey", "jared"
        ]:
            return True
        return False

    wake_ring = deque(maxlen=int(2.0 * SAMPLE_RATE / BLOCK_SIZE))

    with stream:
        print("[Vosk] Listening for wake word ('Jarvis' or press ENTER)...", flush=True)
        while True:
            # Instant Keyboard trigger: Press ENTER or Space in terminal anytime to talk
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in [b'\r', b'\n', b' ']:
                    print("\n[!] Keyboard trigger activated! Listening for command...", flush=True)
                    rec.Reset()
                    _drain_queue()
                    handle_wake(None, ui_pill)
                    wake_ring.clear()
                    rec.Reset()
                    print("\n[Vosk] Listening for 'Jarvis' (or press ENTER)...", flush=True)
                    continue

            data = audio_queue.get()
            wake_ring.append(data)

            heard_text = ""
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                heard_text = res.get("text", "").lower().strip()
                if heard_text:
                    print(f"\n[Vosk heard]: '{heard_text}'", flush=True)
            else:
                part = json.loads(rec.PartialResult())
                heard_text = part.get("partial", "").lower().strip()
                if heard_text:
                    print(f"  [Hearing]: {heard_text}   ", end="\r", flush=True)

            if heard_text and is_wake_word(heard_text):
                print(f"\n[!] Wake word triggered: '{heard_text}'", flush=True)
                rec.Reset()

                wake_audio = b"".join(wake_ring)
                _drain_queue()

                handle_wake(wake_audio, ui_pill)

                wake_ring.clear()
                rec.Reset()
                print("\n[Vosk] Listening for 'Jarvis' (or press ENTER)...", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nJ.A.R.V.I.S. stopped.")
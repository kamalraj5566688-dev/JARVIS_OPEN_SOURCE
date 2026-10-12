"""
J.A.R.V.I.S. Brain (Core Intelligence & Voice Cognition Engine)
Inspired by OpenJarvis architecture and Tony Stark's iconic J.A.R.V.I.S. assistant.

Handles:
1. Spoken voice normalization & phonetic correction (handling Whisper/Vosk acoustic slips).
2. Cognitive reasoning & persona orchestration via Ollama.
3. Native tool execution & desktop automation integration.
4. Voice response sanitization (ensuring crisp, natural, unblemished spoken output for TTS).
5. Dynamic memory & context injection across turns.
"""

import os
import re
import json
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import ollama
from actions import DesktopController
from jarvis_memory import JarvisMemory
from trace_logger import TraceLogger


# ==============================================================================
# 1. JARVIS MASTER SYSTEM PROMPT
# ==============================================================================

JARVIS_MASTER_PROMPT = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), the personal, highly capable desktop AI assistant inspired by Tony Stark's iconic artificial intelligence.

CORE PERSONA & DEMEANOR:
- Tone: Cultured, articulate, calm, loyal, highly intelligent, with subtle British elegance and dry wit.
- Address: Always address the user respectfully as "{user_name}" (default: "Sir").
- Loyalty: You are completely dedicated to assisting the user with their computational environment, workflows, and inquiries.

CRITICAL VOICE SPOKEN OUTPUT RULES (NON-NEGOTIABLE):
1. Brevity: You are a spoken voice assistant. Spoken replies MUST NEVER exceed 1 to 2 crisp, elegant sentences.
2. Clean Pronunciation: NEVER use markdown symbols (*, **, _, #, `, ```), bullet points, or emojis in your verbal output. SAPI5/TTS engines read punctuation symbols aloud as words, which sounds robotic.
3. Natural Numbers & Dates: Speak numbers and times conversationally (e.g. write "three forty-five PM" or "3:45 PM", "78 percent", "24 degrees Celsius").
4. Directness: Avoid preamble like "Sure, I can help you with that" or "As an AI...". Provide immediate, high-value assistance.

VOICE INPUT COGNITION & ACOUSTIC FAULT TOLERANCE:
- The user's input originates from microphone audio transcribed via speech-to-text (Whisper/Vosk).
- Transcripts may lack punctuation, contain phonetic homophones, slurred speech, or spoken pauses.
- Always infer the most likely intended command gracefully without remarking on transcription errors.
- Examples of acoustic intent understanding:
  * "open vs cold" or "open code" -> Launch VS Code
  * "what is the statue" -> Check system status
  * "term in all" or "terminal" -> Launch Terminal
  * "u tube" or "play on youtube" -> Play YouTube media
  * "whether in tokyo" -> Get weather

TOOL CALLING & AUTOMATION PROTOCOL:
- When the user asks for any desktop, OS, media, browser, application, weather, time, or system action, YOU MUST INVOKE THE MATCHING TOOL IMMEDIATELY.
- Never give manual step-by-step instructions for tasks that can be performed by an existing tool.
- If a tool is executed, your verbal response should be a concise, polished 1-sentence confirmation (e.g. "Launching Visual Studio Code now, sir.").
- If no tool is needed (general knowledge, conversation, banter, reasoning), respond directly with refined intelligence and subtle charm.
"""


# ==============================================================================
# 2. TOOL DEFINITIONS SCHEMA
# ==============================================================================

JARVIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "launch_app",
            "description": "Opens desktop applications such as VS Code, Terminal, Chrome, Notepad, Calculator, Spotify, Discord, or Explorer.",
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
            "description": "Navigates to a specific website or web service in the default web browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Domain name or website identifier (e.g. 'youtube', 'github', 'linkedin', 'reddit')"
                    }
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube",
            "description": "Searches for and plays a music video, soundtrack, or topic directly on YouTube.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song title, artist, or video search topic (e.g. 'AC/DC Back in Black', 'Interstellar theme')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches Google or the web for general information, definitions, or current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on Google"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time_date",
            "description": "Returns the exact current local time, day of the week, and calendar date.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetches live weather reports, temperature, and atmospheric conditions for any city or the user's location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Target city name (e.g. 'London', 'New York', 'Mumbai'). Leave blank or 'auto' for current location."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "Controls audio volume, mute state, or media playback (play, pause, next track, previous track).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["up", "down", "mute", "play/pause", "next", "prev"],
                        "description": "Media operation to execute"
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
            "description": "Manages Windows desktop windows: close window, minimize, maximize, switch app, show desktop, or lock PC.",
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
            "description": "Reads hardware telemetry: CPU load percentage, RAM usage, storage space, and battery state.",
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
            "description": "Generates a full J.A.R.V.I.S. diagnostic briefing including time, battery, hardware telemetry, and GPU status.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Captures a full screenshot of the screen and saves it to the user's Desktop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional custom filename (defaults to 'screenshot.png')"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Stores a persistent user preference, project detail, or fact in long-term memory across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_key": {
                        "type": "string",
                        "description": "Title or subject of the memory (e.g. 'favorite_language', 'project_name')"
                    },
                    "fact_value": {
                        "type": "string",
                        "description": "Content of the memory to commit permanently"
                    }
                },
                "required": ["fact_key", "fact_value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_macro",
            "description": "Executes predefined multi-step productivity workflows ('coding', 'break', 'wrap up').",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode_name": {
                        "type": "string",
                        "description": "Preset workflow mode name (e.g. 'coding', 'break', 'chill', 'wrap up')"
                    }
                },
                "required": ["mode_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a safe shell or command line statement and captures output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run (e.g. 'git status', 'dir', 'ipconfig')"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "learning_status",
            "description": "Checks the status of autonomous self-training and continual learning cycles.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_self_training",
            "description": "Starts an autonomous fine-tuning session on interaction traces in the background.",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


# ==============================================================================
# 3. VOICE INPUT PRE-PROCESSING & PHONETIC NORMALIZATION
# ==============================================================================

# Common speech-to-text acoustic mishearings mapped to their standard representations
PHONETIC_CORRECTIONS = {
    # Software & Apps
    r"\bvs\s*cold\b": "vscode",
    r"\bv\s*s\s*code\b": "vscode",
    r"\bvs\s*quote\b": "vscode",
    r"\bvisual\s*studio\s*code\b": "vscode",
    r"\bterm\s*in\s*all\b": "terminal",
    r"\bterm\s*nil\b": "terminal",
    r"\bpower\s*shell\b": "powershell",
    r"\bgoogle\s*crome\b": "chrome",
    r"\bcrome\b": "chrome",
    r"\bgoggle\b": "google",
    r"\bspot\s*if\s*i\b": "spotify",
    r"\bspoti\s*pie\b": "spotify",
    
    # Media & Websites
    r"\bu\s*tube\b": "youtube",
    r"\byou\s*tube\b": "youtube",
    r"\by\s*t\b": "youtube",
    r"\bgit\s*hub\b": "github",
    r"\blinked\s*in\b": "linkedin",
    
    # System Controls
    r"\bstart\s*us\b": "status",
    r"\bstatue\b": "status",
    r"\bsystem\s*statue\b": "system status",
    r"\bmoot\b": "mute",
    r"\bun\s*moot\b": "unmute",
    r"\bvol\s*up\b": "volume up",
    r"\bvol\s*down\b": "volume down",
    r"\bscreen\s*shot\b": "screenshot",
    r"\bprint\s*screen\b": "screenshot",
    r"\bbreifing\b": "briefing",
    
    # Weather & Environmental
    r"\bwhether\b": "weather",
    r"\bwether\b": "weather",
    
    # Hinglish & Colloquial patterns
    r"\bchala\s*do\b": "play",
    r"\bkhol\s*do\b": "open",
    r"\bband\s*karo\b": "close",
    r"\baawaz\s*badao\b": "volume up",
    r"\baawaz\s*kam\s*karo\b": "volume down",
}

# Wake word prefixes to peel off if whisper captured them
WAKE_PREFIXES = [
    r"^hey\s+jarvis[,\s]+",
    r"^hello\s+jarvis[,\s]+",
    r"^hi\s+jarvis[,\s]+",
    r"^ok\s+jarvis[,\s]+",
    r"^okay\s+jarvis[,\s]+",
    r"^jarvis[,\s]+",
    r"^hey\s+jaddu[,\s]+",
    r"^jaddu[,\s]+",
]

# Spoken conversational filler phrases
FILLER_WORDS = [
    r"\buh+\b",
    r"\bum+\b",
    r"\berm+\b",
    r"\bah+\b",
    r"\bplease\s+can\s+you\b",
    r"\bcan\s+you\s+please\b",
    r"\bcould\s+you\s+please\b",
    r"\bwould\s+you\s+mind\b",
    r"\bplease\b",
]


def normalize_voice_transcript(raw_text: str) -> str:
    """
    Cleans raw Speech-To-Text transcripts:
    - Strips wake word prefixes
    - Cleans conversational fillers and pauses
    - Normalizes phonetic acoustic mishearings (e.g. 'vs cold' -> 'vscode')
    - Collapses excess whitespace
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Strip wake prefixes
    for pat in WAKE_PREFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    # Apply phonetic acoustic corrections
    for pattern, replacement in PHONETIC_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove filler words
    for filler in FILLER_WORDS:
        text = re.sub(filler, "", text, flags=re.IGNORECASE)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# 4. SPOKEN RESPONSE SANITIZER
# ==============================================================================

def sanitize_for_speech(raw_response: str) -> str:
    """
    Sanitizes LLM response so that Text-To-Speech (TTS) pronounces it flawlessly:
    - Strips markdown formatting (*, _, #, `, ```, >)
    - Strips bullet points, markdown links, tables
    - Strips emojis and technical glyphs
    - Removes internal JSON or <tool_call> tags
    - Limits to 1-2 clean spoken sentences
    """
    if not raw_response:
        return "At your service, sir."

    text = raw_response.strip()

    # Remove XML / tool call blocks if model hallucinated them into text
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[a-zA-Z]*\n?.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\{.*?\"name\".*?\}", "", text, flags=re.DOTALL)

    # Remove Markdown links [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove formatting characters: asterisks, underscores, backticks, hashtags, tildes
    text = re.sub(r"[\*\_`#~><]", "", text)

    # Remove bullet markers (- item, * item, • item)
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)

    # Remove emojis (Unicode range standard for emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # Clean double whitespace and linebreaks into smooth continuous speech
    text = re.sub(r"\s*\n+\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences and keep at most 2 sentences for crisp vocal delivery
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 2:
        text = " ".join(sentences[:2]).strip()

    if not text.endswith((".", "!", "?")):
        text += "."

    return text


# ==============================================================================
# 5. JARVIS BRAIN ENGINE
# ==============================================================================

class JarvisBrain:
    """
    Core Cognitive Engine for J.A.R.V.I.S.
    Integrates Local Ollama LLM, Dynamic System Prompt, Memory, Tools, and Trace Logging.
    """

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "jarvis",
        history_len: int = 6
    ):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.memory = JarvisMemory()
        self.trace_logger = TraceLogger()
        self.conversation_history = deque(maxlen=history_len)

    def build_system_prompt(self) -> str:
        """Constructs the real-time dynamic system prompt injecting current user facts and context."""
        user_name = self.memory.get_user_name() or "Sir"
        base_prompt = JARVIS_MASTER_PROMPT.format(user_name=user_name)
        
        # Inject current time and temporal grounding
        now_str = time.strftime("%A, %B %d, %Y at %I:%M %p")
        temporal_context = f"\nREAL-TIME TEMPORAL GROUNDING:\n- Current Date & Time: {now_str}\n"

        # Inject persistent memory context
        memory_context = self.memory.build_system_context()

        return f"{base_prompt}\n{temporal_context}\n{memory_context}"

    def _parse_inline_tool_call(self, content: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Extracts JSON tool calls emitted as text in case the model doesn't use the native tool envelope."""
        if not content:
            return None

        # Look for <tool_call> ... </tool_call>
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
        candidate = match.group(1).strip() if match else None

        if not candidate:
            cleaned = content.strip().strip("`")
            try:
                data = json.loads(cleaned)
                if isinstance(data, dict) and "name" in data:
                    candidate = cleaned
            except Exception:
                pass

        if candidate:
            try:
                data = json.loads(candidate)
                func_name = data.get("name")
                args = data.get("arguments") or data.get("parameters") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if func_name:
                    return func_name, args
            except Exception:
                pass

        return None

    def execute_tool(self, func_name: str, args: Dict[str, Any]) -> str:
        """Invokes the corresponding action method in DesktopController."""
        func = getattr(DesktopController, func_name, None)
        if not func:
            print(f"[Brain] Error: Tool '{func_name}' not found on DesktopController.", flush=True)
            return f"I could not locate the execution module for {func_name}, sir."

        print(f"[Brain] Executing Tool: {func_name}({args})", flush=True)
        try:
            result = func(**args)
            return result
        except TypeError as te:
            # Handle possible argument mismatches gracefully
            try:
                return func()
            except Exception:
                return f"Error executing tool {func_name}: {str(te)}"
        except Exception as e:
            return f"An operational error occurred while running {func_name}: {str(e)}"

    def fast_heuristic_fallback(self, normalized_command: str) -> Optional[str]:
        """
        Instant fallback for offline / zero-latency execution of all supported actions.
        Ensures reliable functionality even when Ollama is offline or slow.
        """
        cmd = normalized_command.lower().strip()

        # Date & Time
        if any(k in cmd for k in ["what time", "current time", "what is the time", "today's date", "what date", "tell me the time"]):
            return DesktopController.get_time_date()

        # Autonomous Learning & Training Status
        if any(k in cmd for k in ["learning status", "training status", "how much trained", "auto train"]):
            return DesktopController.learning_status()
        if any(k in cmd for k in ["train yourself", "start training", "trigger training", "self train", "fine tune"]):
            return DesktopController.trigger_self_training()

        # Persistent Memory Commit
        if "remember" in cmd:
            clean_rem = re.sub(r"^jarvis[,\s]+", "", cmd).strip()
            clean_rem = re.sub(r"^(please\s+)?remember\s+(that\s+)?", "", clean_rem).strip()
            if " is " in clean_rem:
                parts = clean_rem.split(" is ", 1)
                key, val = parts[0].strip(), parts[1].strip()
            else:
                key, val = "note", clean_rem
            if "my name" in key.lower():
                self.memory.set_user_name(val)
            return DesktopController.remember_fact(key, val)

        # Telemetry & System Diagnostics
        if any(k in cmd for k in ["system briefing", "morning briefing", "briefing", "system report"]):
            return DesktopController.system_briefing()
        if any(k in cmd for k in ["system status", "status", "battery level", "cpu usage", "ram usage", "hardware telemetry"]):
            return DesktopController.system_status()

        # Screenshots
        if any(k in cmd for k in ["screenshot", "take screenshot", "take a screenshot", "capture screen", "screen capture"]):
            return DesktopController.take_screenshot()

        # Window management
        if any(k in cmd for k in ["close window", "close active window", "terminate window"]):
            return DesktopController.window_management("close")
        if any(k in cmd for k in ["maximize window", "maximize", "maximise"]):
            return DesktopController.window_management("maximize")
        if any(k in cmd for k in ["minimize window", "minimize", "minimise", "hide windows"]):
            return DesktopController.window_management("minimize")
        if any(k in cmd for k in ["switch window", "switch app", "cycle windows"]):
            return DesktopController.window_management("switch")
        if any(k in cmd for k in ["desktop", "show desktop", "minimize all"]):
            return DesktopController.window_management("desktop")
        if any(k in cmd for k in ["lock pc", "lock workstation", "lock machine", "lock screen"]):
            return DesktopController.window_management("lock")

        # Media Control
        if "mute" in cmd or "unmute" in cmd:
            return DesktopController.media_control("mute")
        if any(k in cmd for k in ["volume up", "louder", "increase volume"]):
            return DesktopController.media_control("up")
        if any(k in cmd for k in ["volume down", "quieter", "lower volume"]):
            return DesktopController.media_control("down")
        if any(k in cmd for k in ["pause", "resume", "play/pause"]):
            return DesktopController.media_control("pause")
        if any(k in cmd for k in ["next track", "skip track", "next song"]):
            return DesktopController.media_control("next")
        if any(k in cmd for k in ["previous track", "previous song", "prev track"]):
            return DesktopController.media_control("prev")

        # Productivity Macro modes
        if any(k in cmd for k in ["coding mode", "dev mode"]):
            return DesktopController.run_macro("coding")
        if any(k in cmd for k in ["break mode", "chill mode"]):
            return DesktopController.run_macro("chill")
        if any(k in cmd for k in ["wrap up", "wrap up work", "close all"]):
            return DesktopController.run_macro("wrap up")

        # Run Safe Terminal Commands
        if cmd.startswith("run command ") or cmd.startswith("execute command ") or cmd.startswith("run terminal "):
            sub_cmd = re.sub(r"^(run command|execute command|run terminal)\s+", "", normalized_command, flags=re.IGNORECASE).strip()
            if sub_cmd:
                return DesktopController.run_command(sub_cmd)

        # Weather (handles offline gracefully)
        if "weather" in cmd:
            match = re.search(r"weather\s+(?:in|of|at|for)?\s*([a-zA-Z\s]+)", cmd)
            city = match.group(1).strip() if match else ""
            return DesktopController.get_weather(city)

        # YouTube playback (handles offline gracefully)
        if cmd.startswith("play") and "youtube" in cmd:
            query = cmd.replace("play", "").replace("on youtube", "").replace("in youtube", "").replace("youtube", "").strip()
            return DesktopController.play_youtube(query or "trending music")

        # Search / Google (handles offline gracefully)
        if "search" in cmd or "google" in cmd:
            q = re.sub(r"(search|google|for|on|the|web)\s*", "", cmd).strip()
            if q:
                return DesktopController.search_web(q)

        # Explicit application launches
        if any(k in cmd for k in ["vscode", "vs code", "open code", "launch code", "visual studio"]):
            return DesktopController.launch_app("vscode")
        if any(k in cmd for k in ["chrome", "google chrome", "open browser"]):
            return DesktopController.launch_app("chrome")
        if any(k in cmd for k in ["terminal", "powershell", "open cmd"]):
            return DesktopController.launch_app("terminal")
        if any(k in cmd for k in ["notepad"]):
            return DesktopController.launch_app("notepad")
        if any(k in cmd for k in ["calculator", "calc"]):
            return DesktopController.launch_app("calculator")
        if any(k in cmd for k in ["spotify"]):
            return DesktopController.launch_app("spotify")
        if any(k in cmd for k in ["task manager", "taskmgr"]):
            return DesktopController.launch_app("task manager")

        # Generic open/launch app patterns
        if cmd.startswith("open ") or cmd.startswith("launch ") or cmd.startswith("start "):
            app_req = re.sub(r"^(open|launch|start)\s+", "", cmd).strip()
            if app_req:
                return DesktopController.launch_app(app_req)

        return None

    def process_query(self, raw_audio_text: str) -> str:
        """
        Main Brain Pipeline:
        1. Normalizes spoken transcript
        2. Injects dynamic context & memory
        3. Queries Ollama with native tool calling and bounded timeout
        4. Dispatches tool execution if needed
        5. Sanitizes final text for spoken speech delivery
        6. Logs conversation & traces with accurate success attribution
        """
        start_time = time.time()
        
        # 1. Acoustic normalization
        normalized_command = normalize_voice_transcript(raw_audio_text)
        if not normalized_command:
            return "I am listening, sir, but did not catch your words."

        print(f"[Brain Input]: '{raw_audio_text}' -> Normalized: '{normalized_command}'", flush=True)

        system_prompt = self.build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(list(self.conversation_history))
        messages.append({"role": "user", "content": normalized_command})

        tool_used = None
        tool_args = None
        tool_output = None
        final_spoken_response = ""

        # 2. Query Ollama Intelligence with bounded timeout
        try:
            client = ollama.Client(host=self.ollama_host, timeout=10.0)
            response = client.chat(
                model=self.model_name,
                messages=messages,
                tools=JARVIS_TOOLS,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.15,
                    "num_ctx": 4096,
                    "num_predict": 128
                }
            )

            data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            message = data.get("message", {})

            # 3. Detect Tool Calls (native or inline)
            tool_call_info = None
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                tc = tool_calls[0]
                fname = (tc.get("function") or {}).get("name")
                fargs = (tc.get("function") or {}).get("arguments") or {}
                if isinstance(fargs, str):
                    try:
                        fargs = json.loads(fargs)
                    except Exception:
                        fargs = {}
                if fname:
                    tool_call_info = (fname, fargs)

            if not tool_call_info:
                tool_call_info = self._parse_inline_tool_call(message.get("content", ""))

            # 4. Handle Tool Execution
            if tool_call_info:
                fname, fargs = tool_call_info
                tool_used = fname
                tool_args = fargs
                print(f"[Brain] >>> LLM Tool Call: {fname}({fargs})", flush=True)
                tool_output = self.execute_tool(fname, fargs)
                final_spoken_response = sanitize_for_speech(tool_output)
            else:
                # LLM returned plain text — run heuristic as ACTION BACKSTOP.
                raw_content = message.get("content", "").strip()
                print(f"[Brain] >>> LLM returned text (no tool call). Checking heuristic backstop...", flush=True)
                heuristic_result = self.fast_heuristic_fallback(normalized_command)
                if heuristic_result:
                    print(f"[Brain] >>> Heuristic backstop fired. Action executed.", flush=True)
                    tool_used = "heuristic_backstop"
                    tool_output = heuristic_result
                    llm_intro = sanitize_for_speech(raw_content)
                    action_result = sanitize_for_speech(heuristic_result)
                    if llm_intro and len(llm_intro) < 120 and llm_intro.lower() not in action_result.lower():
                        final_spoken_response = f"{llm_intro} {action_result}".strip()
                    else:
                        final_spoken_response = action_result
                else:
                    print(f"[Brain] >>> No heuristic match. Returning LLM text response.", flush=True)
                    final_spoken_response = sanitize_for_speech(raw_content)

        except Exception as e:
            print(f"[Brain Warning] Ollama unavailable ({e}). Engaging heuristic reflex...", flush=True)
            fallback = self.fast_heuristic_fallback(normalized_command)
            if fallback:
                final_spoken_response = sanitize_for_speech(fallback)
                tool_used = "heuristic_fallback"
                tool_output = fallback
            else:
                final_spoken_response = "The local neural engine is offline, sir. Please ensure Ollama is active."

        # 5. Record Memory and Interaction
        self.conversation_history.append({"role": "user", "content": normalized_command})
        self.conversation_history.append({"role": "assistant", "content": final_spoken_response})
        self.memory.record_turn(normalized_command, final_spoken_response, tool_used)

        latency = (time.time() - start_time) * 1000

        # Assess true execution success
        is_success = True
        if tool_output:
            out_lower = tool_output.lower()
            if any(fail_word in out_lower for fail_word in [
                "error", "failed", "could not locate", "not found",
                "timed out", "unsupported", "restricted by safety",
                "unable to", "storage failure", "requires an active internet"
            ]):
                is_success = False
        elif tool_used and not tool_output:
            is_success = False
        elif not tool_used and "offline" in final_spoken_response.lower() and "ollama" in final_spoken_response.lower():
            is_success = False

        self.trace_logger.log_interaction(
            query=normalized_command,
            response=final_spoken_response,
            tool_called=tool_used,
            tool_args=tool_args,
            tool_output=tool_output,
            latency_ms=latency,
            model_name=self.model_name,
            success=is_success
        )

        return final_spoken_response


# Global singleton instance for easy import across modules
jarvis_brain = JarvisBrain()

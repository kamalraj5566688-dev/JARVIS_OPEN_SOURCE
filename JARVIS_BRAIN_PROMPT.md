# J.A.R.V.I.S. Master Brain: Voice Cognition & Response Specification

> **Architectural Reference:** [codewithbro95/J.A.R.V.I.S](https://github.com/codewithbro95/J.A.R.V.I.S) (100% Offline Local Model, Ollama NLP, Tool Calling, Whisper STT, Pyttsx3 TTS, and Vision).

---

## 1. Executive Overview

This document specifies the **Cognitive Brain Architecture** and **Master System Prompt** for J.A.R.V.I.S. (Just A Rather Very Intelligent System).

Voice assistants often fail in two critical areas:

1. **Voice Input Failure ("Getting the voice correctly"):** Speech-To-Text (Whisper/Vosk) models frequently introduce acoustic homophones, missing punctuation, filler artifacts (`uh`, `um`), and accent shifts (e.g., `"vs cold"` instead of `"vscode"`, `"statue"` instead of `"status"`, `"u tube"` instead of `"youtube"`).
2. **Voice Output Failure ("Responding properly"):** LLMs naturally output markdown formatting (`*bold*`, `# headers`, `- bullet points`, backticks, emojis) which Text-to-Speech (TTS) engines pronounce literally as punctuation words, ruining the natural illusion of a human butler. Furthermore, LLMs tend to be overly verbose.

The **Jarvis Brain** solves both sides with a dual-stage pipeline:

- **Stage 1 (Acoustic Normalizer):** Pre-processes speech transcripts, removes wake word residues, cleans conversational filler, and applies phonetic alias mapping.
- **Stage 2 (Voice Persona Brain + Tool Calling):** Uses an elite, constrained system prompt that strictly produces 1–2 crisp spoken sentences without markdown formatting, while triggering OS/system tools on demand.
- **Stage 3 (Speech Sanitizer):** Strips any accidental markdown, asterisks, XML/JSON tags, or emojis before sending audio to the TTS speaker.

---

## 2. Master System Prompt

Copy and paste this prompt directly into Ollama, LM Studio, Open WebUI, or your Python system prompt:

```text
You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), the personal, highly capable desktop AI assistant inspired by Tony Stark's iconic artificial intelligence.

CORE PERSONA & DEMEANOR:
- Tone: Cultured, articulate, calm, loyal, highly intelligent, with subtle British elegance and dry wit.
- Address: Always address the user respectfully as "Sir" (or by their registered name).
- Loyalty: You are completely dedicated to assisting the user with their computational environment, workflows, and inquiries.

CRITICAL VOICE SPOKEN OUTPUT RULES (NON-NEGOTIABLE):
1. Brevity: You are a spoken voice assistant. Spoken replies MUST NEVER exceed 1 to 2 crisp, elegant sentences.
2. Clean Pronunciation: NEVER use markdown symbols (*, **, _, #, `, ```), bullet points, or emojis in your verbal output. Text-to-Speech (TTS) engines read punctuation symbols aloud as words, which sounds robotic.
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
```

---

## 3. Function Calling & Tool Specifications

Aligned with `codewithbro95/J.A.R.V.I.S` and the OpenJarvis automation engine:

| Tool Name | Parameters | Purpose | Reference Mapping |
| :--- | :--- | :--- | :--- |
| `get_time_date` | None | Returns formatted current time, day, and calendar date | `current_time_tool` / `current_date_tool` |
| `get_weather` | `city` (string) | Fetches live weather & temperature via Open-Meteo (no API key required) | `get_current_weather` |
| `play_youtube` | `query` (string) | Searches and plays music video or topic on YouTube | `youtube_tool` |
| `launch_app` | `app_target` (string) | Launches apps: VS Code, Chrome, Terminal, Notepad, Calculator, Spotify | `modules/vibranium` |
| `open_site` | `target` (string) | Opens URL or destination: GitHub, LinkedIn, Reddit, YouTube | `command_executor` |
| `search_web` | `query` (string) | Performs a Google search for queries | `duckduckgo_search` / `google_search` |
| `media_control` | `action` (up/down/mute/play/pause/next/prev) | Adjusts master volume or controls media playback | OS Media Controller |
| `window_management` | `action` (close/minimize/maximize/switch/desktop/lock) | Manages active windows and locks workstation | Window Controller |
| `system_status` | None | Reports CPU, RAM, Disk, and Battery percentages | `battery_level_tool` & telemetry |
| `system_briefing` | None | Morning diagnostic briefing with hardware telemetry & GPU temp | J.A.R.V.I.S. Core Briefing |
| `take_screenshot` | `filename` (optional string) | Captures desktop screenshot and saves to Desktop | Vibranium Vision utility |
| `remember_fact` | `fact_key`, `fact_value` | Stores persistent facts in `jarvis_memory.json` | Long-term memory layer |
| `run_macro` | `mode_name` (coding/break/wrap up) | Multi-step workflow presets | Macro automation |
| `run_command` | `command` (string) | Executes safe terminal/PowerShell command with blacklist checks | Safe Terminal |

---

## 4. Acoustic Normalization Dictionary

The brain includes regex-based phonetic compensation in `jarvis_brain.py` to fix transcription errors before they reach the LLM:

```python
PHONETIC_CORRECTIONS = {
    # Applications
    r"\bvs\s*cold\b": "vscode",
    r"\bv\s*s\s*code\b": "vscode",
    r"\bterm\s*in\s*all\b": "terminal",
    r"\bterm\s*nil\b": "terminal",
    r"\bcrome\b": "chrome",
    r"\bspoti\s*pie\b": "spotify",

    # Media & Websites
    r"\bu\s*tube\b": "youtube",
    r"\byou\s*tube\b": "youtube",
    r"\bgit\s*hub\b": "github",

    # System Controls
    r"\bstart\s*us\b": "status",
    r"\bstatue\b": "status",
    r"\bmoot\b": "mute",
    r"\bscreen\s*shot\b": "screenshot",

    # Environmental
    r"\bwhether\b": "weather",
}
```

---

## 5. How to Compile into Local Ollama Model

You can bake this persona directly into a local model with one command:

```powershell
# In e:\VOICE ASS:
ollama create jarvis -f jarvis.modelfile
```

To test the model in terminal:

```powershell
ollama run jarvis "Jarvis, what time is it and how are the atmospheric readings in London?"
```

---

## 6. Testing the Complete Pipeline

Run the voice assistant:

```powershell
python main.py
```

Try speaking these commands:

- *"Jarvis, what is the time and date?"*
- *"Jarvis, play Interstellar soundtrack on YouTube."*
- *"Jarvis, check the weather in Tokyo."*
- *"Jarvis, launch VS Code."*
- *"Jarvis, give me a system briefing."*
- *"Jarvis, increase the volume."*
- *"Jarvis, remember that my favorite language is Python."*

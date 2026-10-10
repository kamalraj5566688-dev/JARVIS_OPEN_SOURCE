# J.A.R.V.I.S. (OpenJarvis Architecture) Quickstart & Training Guide

This system has been upgraded to match the architecture of Stanford's [OpenJarvis](https://github.com/open-jarvis/OpenJarvis), combining **local Small Language Models (SLMs)**, **OpenJarvis Persistent Memory**, **Interaction Trace Logging**, and a **Closed-Loop QLoRA Training Pipeline** optimized for your **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**.

---

## 1. Quick Start: Launching Jarvis

### Step 1: Ensure Ollama is Running

If you want the full neural intelligence engine active:

1. Open Ollama from your Windows Start Menu (or system tray).
2. To compile your customized Jarvis model into Ollama:

   ```powershell
   ollama create jarvis -f jarvis.modelfile
   ```

### Step 2: Run the Assistant

In your terminal (inside `d:\VOICE ASS` with `.venv` active):

```powershell
python main.py
```

### Step 3: Speak to Jarvis

- Say **"Jarvis"** or **"Hey Jarvis"** (or phonetic variants).
- Jarvis verifies your voice biometrics against [voice_profile.npy](file:///d:/VOICE%20ASS/voice_profile.npy) (ensuring only you can control the desktop).
- Speak your command:
  - *"Jarvis, give me a system briefing."*
  - *"Jarvis, launch VS Code and terminal."*
  - *"Jarvis, what's our CPU, RAM, and GPU status?"*
  - *"Jarvis, remember that my main codebase is in D drive."*
  - *"Jarvis, turn up the volume."*
  - *"Jarvis, minimize all windows."*

---

## 2. How the OpenJarvis Learning & Training Loop Works

Unlike traditional assistants that are static, OpenJarvis improves over time through a **closed-loop**:

```text
[Voice Input] -> [Whisper STT] -> [JARVIS LLM] -> [Tool Call] -> [Execution]
                                                         │
                                                         ▼
                                                [traces.jsonl]
                                                         │
                                                         ▼
                                            [train_jarvis.py --generate]
                                                         │
                                                         ▼
                                            [train_jarvis.py --train]
                                            (QLoRA on RTX 3050 4GB GPU)
                                                         │
                                                         ▼
                                            [Updated Model / Adapter]
```

### 1. Automatic Trace Logging ([trace_logger.py](file:///d:/VOICE%20ASS/trace_logger.py))

Every interaction, spoken command, tool executed, and latency measurement is automatically saved to [traces.jsonl](file:///d:/VOICE%20ASS/traces.jsonl).

### 2. Generate Training Datasets

Compile your live usage history into a Supervised Fine-Tuning (SFT) dataset:

```powershell
python train_jarvis.py --generate
```

This generates [dataset_jarvis_sft.json](file:///d:/VOICE%20ASS/dataset_jarvis_sft.json), combining your real traces with curated JARVIS instruction data.

### 3. Fine-Tune Your Personal Model (QLoRA)

To fine-tune the model parameters on your RTX 3050:

1. Install training dependencies into your virtual environment:

   ```powershell
   .\.venv\Scripts\pip.exe install -r requirements-training.txt
   ```

2. Run training:

   ```powershell
   python train_jarvis.py --train
   ```

3. Re-compile into Ollama:

   ```powershell
   python train_jarvis.py --deploy
   ```

---

## 3. Persistent Memory ([jarvis_memory.py](file:///d:/VOICE%20ASS/jarvis_memory.py))

Jarvis maintains permanent memory in [jarvis_memory.json](file:///d:/VOICE%20ASS/jarvis_memory.json):

- Remembers who you are (e.g. "Sir", your projects, your preferences).
- Preserves context across computer reboots.
- You can tell Jarvis: *"Jarvis, remember that my preferred browser is Chrome"* and it will store it indefinitely.

---

## 4. Key Subsystem Files

- [main.py](file:///d:/VOICE%20ASS/main.py): Central orchestrator loop with Vosk wake spotting, Whisper STT, voice verification, and tool dispatch.
- [actions.py](file:///d:/VOICE%20ASS/actions.py): Universal desktop automation tools (telemetry, GPU sensors, app launching, macros, memory).
- [jarvis_memory.py](file:///d:/VOICE%20ASS/jarvis_memory.py): OpenJarvis long-term memory engine.
- [trace_logger.py](file:///d:/VOICE%20ASS/trace_logger.py): OpenJarvis trace logger for continuous improvement.
- [train_jarvis.py](file:///d:/VOICE%20ASS/train_jarvis.py): Synthetic dataset generation and QLoRA fine-tuning pipeline.
- [jarvis.modelfile](file:///d:/VOICE%20ASS/jarvis.modelfile): Ollama configuration for the JARVIS persona.
- [audio_io.py](file:///d:/VOICE%20ASS/audio_io.py): Text-to-Speech (JARVIS cadence) and Whisper STT.

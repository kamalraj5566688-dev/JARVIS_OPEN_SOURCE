# Release v1.0.0 — J.A.R.V.I.S. Open Source (Offline Edition)

## 🚀 What's New in v1.0.0

This marks the official initial open-source release of **J.A.R.V.I.S. (Offline Edition)**, a 100% on-device, privacy-first personal desktop voice assistant inspired by Iron Man's JARVIS and Stanford's **OpenJarvis** agent architecture.

---

### 🌟 Key Highlights

- **🔒 100% Offline & Private:** Zero cloud dependencies, zero latency bottlenecks, zero external telemetry.
- **🎙️ Biometric Speaker Authentication:** Resemblyzer acoustic voice biometrics verify the owner's voice before executing commands, rejecting unauthorized users.
- **⚡ Fast Wake-Word Spotting:** Low-latency continuous listening via Vosk Kaldi engine for *"Jarvis"* and *"Hey Jarvis"*.
- **🧠 Local LLM Reasoning:** Direct integration with Ollama (`qwen2.5:1.5b`, `qwen2.5:3b`, `llama3.2:3b`) with native JSON function calling.
- **💾 OpenJarvis Persistent Memory:** Long-term memory store (`jarvis_memory.json`) preserves user identity, project paths, and dynamic facts across reboots.
- **🔄 Autonomous Continuous Self-Training:** Built-in background daemon (`auto_trainer.py`) monitors interaction traces (`traces.jsonl`) and automatically triggers incremental QLoRA fine-tuning on your local GPU.
- **🖥️ Universal Desktop Automation:** Controls audio volume, media, windows, launches applications, executes safe shell commands, and captures screenshots.
- **📊 Live Hardware & GPU Telemetry:** Pulls real-time CPU %, RAM %, Battery %, and live NVIDIA RTX GPU load and temperature via `nvidia-smi`.
- **🎨 Translucent Floating HUD:** Borderless Tkinter status pill displaying real-time assistant states (`IDLE`, `LISTENING`, `VERIFYING`, `THINKING`, `SPEAKING`).

---

### 📦 Files & Modules Included

- `main.py` — Central voice loop and orchestrator.
- `actions.py` — Desktop automation and hardware telemetry suite.
- `audio_io.py` — Faster-whisper STT and calibrated pyttsx3 vocal feedback.
- `speaker_verifier.py` — Resemblyzer voice biometrics authentication.
- `enroll_voice.py` — Owner voice enrollment CLI tool.
- `jarvis_memory.py` — OpenJarvis persistent memory engine.
- `trace_logger.py` — OpenJarvis interaction trace collector.
- `train_jarvis.py` — SFT dataset generator and GPU QLoRA fine-tuning pipeline.
- `auto_trainer.py` — Autonomous background continual self-training daemon.
- `jarvis.modelfile` — Ollama JARVIS persona and hyperparameters.
- `overlay.py` — Floating status pill HUD.
- `setup_models.py` — Automated offline Vosk speech model installer.

---

### 🛠️ Quick Installation

```powershell
git clone https://github.com/kamalraj5566688-dev/JARVIS_OPEN_SOURCE.git
cd JARVIS_OPEN_SOURCE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_models.py
python enroll_voice.py
ollama create jarvis -f jarvis.modelfile
python main.py
```

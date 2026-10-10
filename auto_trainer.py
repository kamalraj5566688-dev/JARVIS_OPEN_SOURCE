"""
auto_trainer.py
===============
OpenJarvis Autonomous Self-Training & Continuous Learning Daemon.

Monitors interaction traces and autonomously triggers background fine-tuning
cycles without interrupting user voice commands or desktop operations.
"""

import os
import json
import time
import threading
import subprocess
from typing import Dict, Any, Optional

STATE_FILE = os.path.join(os.path.dirname(__file__), "training_state.json")
TRACES_FILE = os.path.join(os.path.dirname(__file__), "traces.jsonl")
DEFAULT_THRESHOLD = 20  # Train every 20 new interactions


class JarvisAutoTrainer:
    """
    Autonomous Continuous Self-Training Engine.
    Executes closed-loop learning from user voice traces automatically in the background.
    """
    def __init__(self, threshold: int = DEFAULT_THRESHOLD, state_path: str = STATE_FILE):
        self.threshold = threshold
        self.state_path = state_path
        self._lock = threading.Lock()
        self.is_training = False
        self.state = self._load_state()
        if not os.path.exists(self.state_path):
            self._save_state()

    def _load_state(self) -> Dict[str, Any]:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_cycles": 1 if os.path.exists(os.path.join(os.path.dirname(__file__), "jarvis_lora_adapter")) else 0,
            "last_trained_timestamp": time.time(),
            "last_trained_trace_count": self._count_current_traces(),
            "status": "idle",
            "history": []
        }

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AutoTrainer] Error saving state: {e}")

    def _count_current_traces(self) -> int:
        if not os.path.exists(TRACES_FILE):
            return 0
        try:
            with open(TRACES_FILE, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def get_status_summary(self) -> str:
        """Returns a concise spoken status summary for voice queries."""
        current_traces = self._count_current_traces()
        last_count = self.state.get("last_trained_trace_count", 0)
        new_traces = max(0, current_traces - last_count)
        remaining = max(0, self.threshold - new_traces)
        cycles = self.state.get("total_cycles", 0)

        if self.is_training:
            return f"I am currently running an autonomous fine-tuning cycle in the background, sir."
        
        return (
            f"Autonomous learning is active, sir. I have completed {cycles} evolution cycles. "
            f"I have logged {new_traces} new interaction traces, and will automatically trigger the next "
            f"training run in {remaining} commands."
        )

    def on_interaction(self, voice_notifier: Optional[Any] = None) -> None:
        """
        Called after every voice command is completed.
        Checks if the trace threshold is reached and triggers background self-training.
        """
        current_traces = self._count_current_traces()
        last_count = self.state.get("last_trained_trace_count", 0)
        new_traces = current_traces - last_count

        if new_traces >= self.threshold and not self.is_training:
            print(f"\n[AutoTrainer] Threshold reached ({new_traces}/{self.threshold} new traces).")
            print("[AutoTrainer] Launching autonomous background self-training cycle...", flush=True)
            self.trigger_background_training(voice_notifier=voice_notifier)

    def trigger_background_training(self, force: bool = False, voice_notifier: Optional[Any] = None) -> bool:
        """
        Spawns a detached background thread to execute the full training loop.
        """
        with self._lock:
            if self.is_training:
                print("[AutoTrainer] Self-training is already running in background.")
                return False
            self.is_training = True
            self.state["status"] = "training"
            self._save_state()

        worker = threading.Thread(
            target=self._run_training_pipeline,
            args=(voice_notifier,),
            daemon=True
        )
        worker.start()
        return True

    def _run_training_pipeline(self, voice_notifier: Optional[Any] = None) -> None:
        """Internal worker executing dataset generation, LoRA fine-tuning, and deployment."""
        try:
            start_time = time.time()
            current_traces = self._count_current_traces()
            print("\n[AutoTrainer] === Phase 1: Synthesizing SFT Dataset ===", flush=True)

            script_path = os.path.join(os.path.dirname(__file__), "train_jarvis.py")
            python_exe = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
            if not os.path.exists(python_exe):
                python_exe = "python"

            # 1. Regenerate training dataset
            subprocess.run([python_exe, script_path, "--generate"], check=True)

            print("[AutoTrainer] === Phase 2: Autonomous LoRA Fine-Tuning ===", flush=True)
            # 2. Run LoRA fine-tuning
            subprocess.run([python_exe, script_path, "--train"], check=True)

            print("[AutoTrainer] === Phase 3: Hot-Deploying Model to Ollama ===", flush=True)
            # 3. Deploy updated weights
            subprocess.run([python_exe, script_path, "--deploy"], check=False)

            duration_min = (time.time() - start_time) / 60.0
            cycles = self.state.get("total_cycles", 0) + 1

            self.state["total_cycles"] = cycles
            self.state["last_trained_timestamp"] = time.time()
            self.state["last_trained_trace_count"] = current_traces
            self.state["status"] = "idle"
            self.state.setdefault("history", []).append({
                "cycle": cycles,
                "timestamp": time.time(),
                "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "traces_count": current_traces,
                "duration_minutes": round(duration_min, 2)
            })
            self._save_state()

            print(f"\n[AutoTrainer] [+] Self-Training Cycle {cycles} completed in {duration_min:.1f}m!\n", flush=True)

            if voice_notifier and hasattr(voice_notifier, "speak"):
                try:
                    voice_notifier.speak(
                        f"Autonomous self-training cycle {cycles} is complete, sir. "
                        f"Subsystems have been updated and enhanced."
                    )
                except Exception:
                    pass

        except Exception as e:
            print(f"[AutoTrainer] [!] Error during background self-training: {e}", flush=True)
            self.state["status"] = f"error: {str(e)}"
            self._save_state()

        finally:
            with self._lock:
                self.is_training = False


# Global singleton instance
auto_trainer = JarvisAutoTrainer()

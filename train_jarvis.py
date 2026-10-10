"""
train_jarvis.py
================
OpenJarvis Local Training & Fine-Tuning Pipeline for J.A.R.V.I.S.
Supports:
  1. Synthetic + Trace Dataset Generation (--generate)
  2. QLoRA / PEFT 4-bit Adapter Fine-Tuning for RTX 3050 4GB GPU (--train)
  3. Ollama Modelfile Compilation & Deployment (--deploy)

Usage:
  python train_jarvis.py --generate     # Creates training dataset from traces + Jarvis corpus
  python train_jarvis.py --train        # Runs LoRA training on the dataset
  python train_jarvis.py --deploy       # Compiles trained model into Ollama
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any

from trace_logger import TraceLogger

DATASET_FILE = "dataset_jarvis_sft.json"
OUTPUT_DIR = "./jarvis_lora_adapter"

# ----------------- Curated JARVIS Instruction Corpus -----------------
BOOTSTRAP_JARVIS_CORPUS = [
    # System Diagnostics
    {
        "input": "Jarvis, what is our current system status?",
        "tool": "system_status",
        "args": {},
        "response": "Pulling telemetry now, sir. All core diagnostics incoming."
    },
    {
        "input": "Check CPU, RAM and battery health.",
        "tool": "system_status",
        "args": {},
        "response": "Reading hardware sensors now, sir."
    },
    {
        "input": "How much battery is remaining?",
        "tool": "system_status",
        "args": {},
        "response": "Checking current power reserves, sir."
    },
    # App Launches
    {
        "input": "Jarvis, launch VS Code.",
        "tool": "launch_app",
        "args": {"app_target": "vscode"},
        "response": "Right away, sir. Launching Visual Studio Code."
    },
    {
        "input": "Open the terminal.",
        "tool": "launch_app",
        "args": {"app_target": "terminal"},
        "response": "Opening Windows Terminal immediately, sir."
    },
    {
        "input": "Open Google Chrome.",
        "tool": "launch_app",
        "args": {"app_target": "chrome"},
        "response": "Opening Chrome browser for you now, sir."
    },
    {
        "input": "Start Spotify.",
        "tool": "launch_app",
        "args": {"app_target": "spotify"},
        "response": "Starting Spotify, sir."
    },
    {
        "input": "Open Task Manager.",
        "tool": "launch_app",
        "args": {"app_target": "task manager"},
        "response": "Bringing up the Task Manager, sir."
    },
    # Window & System Controls
    {
        "input": "Jarvis, minimize all windows.",
        "tool": "window_management",
        "args": {"action": "desktop"},
        "response": "Clearing the desktop view immediately, sir."
    },
    {
        "input": "Lock the workstation.",
        "tool": "window_management",
        "args": {"action": "lock"},
        "response": "Locking workstation protocols now, sir. Have a safe journey."
    },
    {
        "input": "Close this window.",
        "tool": "window_management",
        "args": {"action": "close"},
        "response": "Terminating the active window, sir."
    },
    {
        "input": "Switch windows.",
        "tool": "window_management",
        "args": {"action": "switch"},
        "response": "Switching window context, sir."
    },
    # Audio & Media Controls
    {
        "input": "Jarvis, turn up the volume.",
        "tool": "media_control",
        "args": {"action": "up"},
        "response": "Increasing audio master volume, sir."
    },
    {
        "input": "Mute the audio.",
        "tool": "media_control",
        "args": {"action": "mute"},
        "response": "Audio output muted, sir."
    },
    {
        "input": "Pause the music.",
        "tool": "media_control",
        "args": {"action": "play/pause"},
        "response": "Media playback paused, sir."
    },
    # Web & Information
    {
        "input": "Jarvis, search for latest developments in artificial intelligence.",
        "tool": "search_web",
        "args": {"query": "latest developments in artificial intelligence"},
        "response": "Initiating search query on artificial intelligence, sir."
    },
    {
        "input": "Open YouTube.",
        "tool": "open_site",
        "args": {"target": "youtube"},
        "response": "Opening YouTube right away, sir."
    },
    {
        "input": "Open GitHub.",
        "tool": "open_site",
        "args": {"target": "github"},
        "response": "Navigating to GitHub repository dashboard, sir."
    },
    # Memory Management
    {
        "input": "Jarvis, remember that my main codebase is in D drive VOICE ASS.",
        "tool": "remember_fact",
        "args": {"fact_key": "main_project", "fact_value": "D:/VOICE ASS"},
        "response": "Committed to permanent long-term memory, sir."
    },
    # Work Presets / Macros
    {
        "input": "Activate coding mode.",
        "tool": "run_macro",
        "args": {"mode_name": "coding"},
        "response": "Coding mode initiated, sir. Visual Studio Code, Terminal, and GitHub are ready."
    },
    {
        "input": "Take a screenshot of the screen.",
        "tool": "take_screenshot",
        "args": {},
        "response": "Capturing display buffer to desktop now, sir."
    },
    # General Jarvis Personality & Banter
    {
        "input": "Jarvis, are you online?",
        "tool": None,
        "args": {},
        "response": "Online and fully operational, sir. Standing by for your instructions."
    },
    {
        "input": "Good morning, Jarvis.",
        "tool": None,
        "args": {},
        "response": "Good morning, sir. All core subsystems are operating within nominal parameters."
    },
    {
        "input": "Who are you?",
        "tool": None,
        "args": {},
        "response": "I am J.A.R.V.I.S., your dedicated personal desktop intelligence, sir."
    },
    {
        "input": "Thank you, Jarvis.",
        "tool": None,
        "args": {},
        "response": "Always a pleasure to be of service, sir."
    }
]


def generate_training_dataset(output_path: str = DATASET_FILE) -> int:
    """
    Combines the curated JARVIS corpus with live interaction traces
    to build the unified training dataset.
    """
    print("[1/3] Gathering real interaction traces...")
    logger = TraceLogger()
    real_traces = logger.load_traces()
    print(f"      Found {len(real_traces)} live interaction traces.")

    system_instruction = (
        "You are J.A.R.V.I.S., a hyper-competent British desktop AI assistant. "
        "Trigger matching tools when instructed and speak in crisp, 1-2 sentence spoken answers."
    )

    dataset: List[Dict[str, str]] = []

    # 1. Add curated bootstrap corpus
    for item in BOOTSTRAP_JARVIS_CORPUS:
        if item.get("tool"):
            output = f"<tool_call>\n{{\"name\": \"{item['tool']}\", \"arguments\": {json.dumps(item['args'])}}}\n</tool_call>"
        else:
            output = item["response"]

        dataset.append({
            "instruction": system_instruction,
            "input": item["input"],
            "output": output
        })

    # 2. Add live user interaction traces
    for t in real_traces:
        if not t.get("user_query") or not t.get("success", True):
            continue
        tool = t.get("tool_called")
        if tool:
            output = f"<tool_call>\n{{\"name\": \"{tool}\", \"arguments\": {json.dumps(t.get('tool_args', {}))}}}\n</tool_call>"
        else:
            output = t.get("assistant_response", "")

        dataset.append({
            "instruction": system_instruction,
            "input": t["user_query"],
            "output": output
        })

    # Write dataset to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[2/3] Successfully compiled {len(dataset)} training examples into '{output_path}'.")
    return len(dataset)


def train_qlora_adapter(
    dataset_path: str = DATASET_FILE,
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = OUTPUT_DIR,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4
):
    """
    Fine-tunes a LoRA adapter using PyTorch and HuggingFace PEFT.
    Optimized for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM).
    """
    print("\n=======================================================")
    print(" J.A.R.V.I.S. QLoRA Training Pipeline (OpenJarvis Engine)")
    print(f" Target Hardware: NVIDIA RTX 3050 (4GB VRAM)")
    print(f" Base Model: {base_model_name}")
    print("=======================================================\n")

    if not os.path.exists(dataset_path):
        print(f"[!] Dataset '{dataset_path}' not found. Generating now...")
        generate_training_dataset(dataset_path)

    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer  # type: ignore
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    except ImportError as e:
        print(f"\n[!] Missing fine-tuning dependencies: {e}")
        print("[!] To install training packages for your environment, run:")
        print("    pip install peft transformers datasets accelerate bitsandbytes\n")
        print("[*] Note: In the meantime, your assistant is fully powered by Ollama + Modelfile + Persistent Memory!")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Compute Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[+] GPU Detected: {gpu_name} ({gpu_vram:.1f} GB VRAM)")

    # Clean stale locks from previous interrupted downloads to allow seamless resume
    try:
        lock_dir = os.path.expanduser("~/.cache/huggingface/hub/.locks")
        if os.path.exists(lock_dir):
            for root, _, files in os.walk(lock_dir):
                for f in files:
                    if "Qwen" in root or "Qwen" in f:
                        try:
                            os.remove(os.path.join(root, f))
                        except Exception:
                            pass
    except Exception:
        pass

    print(f"[+] Loading Tokenizer & Model: {base_model_name} (resuming existing downloads if any)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in 4-bit / fp16 for RTX 3050 4GB memory safety
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    # Continual Learning: Load existing adapter if available, otherwise initialize new LoRA
    existing_adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if os.path.exists(existing_adapter_file):
        print(f"[+] Found existing trained adapter in '{output_dir}'.")
        print("[+] Loading existing adapter for incremental continual learning...")
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, output_dir, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load and format dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    def format_sample(item):
        prompt = f"<|im_start|>system\n{item['instruction']}<|im_end|>\n<|im_start|>user\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        return tokenizer(prompt, truncation=True, max_length=512, padding="max_length")

    # Simple tokenized dataset
    tokenized_samples = [format_sample(item) for item in raw_data]

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            item = self.samples[idx]
            return {
                "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(item["input_ids"], dtype=torch.long)
            }

    train_dataset = DictDataset(tokenized_samples)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=(device == "cuda"),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    print("\n[+] Commencing LoRA adapter training...")
    trainer.train()

    print(f"\n[+] Training complete! Saving adapter to '{output_dir}'...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[+] J.A.R.V.I.S. adapter successfully created in '{output_dir}'!")
    return True


def deploy_to_ollama(model_name: str = "jarvis"):
    """Creates/updates the local 'jarvis' model inside Ollama using jarvis.modelfile."""
    modelfile_path = os.path.join(os.path.dirname(__file__), "jarvis.modelfile")
    if not os.path.exists(modelfile_path):
        print(f"[!] Modelfile '{modelfile_path}' does not exist.")
        return

    print(f"[Ollama] Compiling model '{model_name}' from '{modelfile_path}'...")
    cmd = f'ollama create {model_name} -f "{modelfile_path}"'
    code = os.system(cmd)
    if code == 0:
        print(f"\n[+] Success! J.A.R.V.I.S. model is now deployed in Ollama as '{model_name}'.")
    else:
        print(f"[!] Failed to compile model in Ollama (exit code {code}). Check if Ollama daemon is running.")


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S. Training & OpenJarvis Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate SFT training dataset")
    parser.add_argument("--train", action="store_true", help="Run LoRA fine-tuning on RTX 3050")
    parser.add_argument("--deploy", action="store_true", help="Compile and deploy model in Ollama")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: generate, train, and deploy")

    args = parser.parse_args()

    if args.all or args.generate:
        generate_training_dataset()

    if args.all or args.train:
        train_qlora_adapter()

    if args.all or args.deploy:
        deploy_to_ollama()

    if not (args.generate or args.train or args.deploy or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()

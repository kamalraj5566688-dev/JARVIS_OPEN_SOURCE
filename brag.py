"""
J.A.R.V.I.S. Brag CLI Tool
Allows running `brag` directly from PowerShell / Command Prompt.
"""
import sys
import os
import subprocess
from pathlib import Path

# Fix Windows console encoding for Unicode / emoji characters
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

WORKSPACE_DIR = Path(__file__).resolve().parent
BRAG_OUTPUT = WORKSPACE_DIR / "brag-output"

def show_help():
    print("""
============================================================
              J.A.R.V.I.S. /brag Launch Suite
============================================================
Usage:
    python brag.py [option]
    ./brag.ps1 [option]
    brag [option]

Options:
    --summary, -s    Display 2-minute LinkedIn video storyboard & hook
    --copy, -c       Print ready-to-post LinkedIn caption & copy to clipboard
    --brief, -b      Display full composition scene-by-scene director's brief
    --plan, -p       Display technical audio, color, and pacing plan
    --render, -r     Run Hyperframes video rendering pipeline (if installed)
    --help, -h       Show this help message
============================================================
""")

def copy_to_clipboard(text: str):
    try:
        process = subprocess.Popen(["clip"], stdin=subprocess.PIPE, text=True)
        process.communicate(input=text)
        print("📋 Copied to Windows clipboard! You can now press Ctrl+V directly on LinkedIn.\n")
    except Exception:
        pass

def show_summary():
    brief_file = BRAG_OUTPUT / "brief.md"
    if brief_file.exists():
        print("\n" + "=" * 60)
        print("🎬 2-MINUTE LINKEDIN DEMO VIDEO STORYBOARD")
        print("=" * 60 + "\n")
        print(brief_file.read_text(encoding="utf-8"))
    else:
        print("[!] brag-output/brief.md not found. Generating...")

def show_copy():
    copy_file = BRAG_OUTPUT / "share-copy.txt"
    if copy_file.exists():
        content = copy_file.read_text(encoding="utf-8")
        print("\n" + "=" * 60)
        print("📱 READY-TO-POST LINKEDIN CAPTION")
        print("=" * 60 + "\n")
        print(content)
        copy_to_clipboard(content)
    else:
        print("[!] brag-output/share-copy.txt not found.")

def show_plan():
    plan_file = BRAG_OUTPUT / "plan.md"
    if plan_file.exists():
        print(plan_file.read_text(encoding="utf-8"))
    else:
        print("[!] brag-output/plan.md not found.")

def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "--summary"
    if arg in ("--help", "-h"):
        show_help()
    elif arg in ("--copy", "-c"):
        show_copy()
    elif arg in ("--plan", "-p"):
        show_plan()
    elif arg in ("--brief", "-b"):
        show_summary()
    elif arg in ("--summary", "-s"):
        show_summary()
        show_copy()
    elif arg in ("--render", "-r"):
        print("[*] Checking Hyperframes renderer...")
        subprocess.run(["npx", "hyperframes", "preview"], shell=True)
    else:
        show_summary()

if __name__ == "__main__":
    main()

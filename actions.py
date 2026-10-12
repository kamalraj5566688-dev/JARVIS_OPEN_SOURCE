import os
import subprocess
import webbrowser
import platform
import shutil
import time
import pyautogui
import psutil

from jarvis_memory import JarvisMemory

# Safe fallback margin: moving mouse to any corner cancels stuck actions
pyautogui.FAILSAFE = True


class DesktopController:
    @staticmethod
    def learning_status() -> str:
        """Reports autonomous self-training and continual learning status."""
        from auto_trainer import auto_trainer
        return auto_trainer.get_status_summary()

    @staticmethod
    def trigger_self_training() -> str:
        """Triggers an autonomous background fine-tuning cycle immediately."""
        from auto_trainer import auto_trainer
        success = auto_trainer.trigger_background_training()
        if success:
            return "Autonomous self-training cycle initiated in the background, sir. I will notify you upon completion."
        return "An autonomous training cycle is already in progress, sir."

    @staticmethod
    def remember_fact(fact_key: str, fact_value: str) -> str:
        """Stores a persistent fact or preference in Jarvis's long-term memory."""
        memory = JarvisMemory()
        return memory.set_fact(fact_key, fact_value)

    @staticmethod
    def system_briefing() -> str:
        """Provides a comprehensive JARVIS telemetry briefing."""
        now_str = time.strftime("%I:%M %p")
        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        
        batt_info = "powered by AC"
        if battery:
            plugged = "charging" if battery.power_plugged else "on battery"
            batt_info = f"battery at {battery.percent}%, {plugged}"

        # GPU detection
        gpu_info = ""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_util, gpu_temp = res.stdout.strip().split(",")
                gpu_info = f", GPU at {gpu_util.strip()}% and {gpu_temp.strip()} degrees Celsius"
        except Exception:
            pass

        return f"Good day, sir. The time is {now_str}. CPU load is at {cpu}%, RAM utilization is at {ram}%, {batt_info}{gpu_info}. All core systems are nominal."

    @staticmethod
    def run_macro(mode_name: str) -> str:
        """Executes combined setup routines for specific work modes."""
        mode = mode_name.lower().strip()
        system = platform.system().lower()

        if "code" in mode or "dev" in mode or "coding" in mode:
            # 1. Open VS Code
            subprocess.Popen("code", shell=True)
            # 2. Open Terminal
            if system == "windows":
                subprocess.Popen("wt.exe", shell=True)
            else:
                subprocess.Popen(["x-terminal-emulator"])
            # 3. Open Browser to GitHub / Dev Docs
            webbrowser.open("https://github.com")
            return "Coding protocol initiated, sir. Visual Studio Code, Terminal, and GitHub are now open."

        elif "chill" in mode or "break" in mode or "music" in mode:
            webbrowser.open("https://music.youtube.com")
            for _ in range(3):
                pyautogui.press("volumedown")
            return "Break mode engaged, sir. Music playback opened and sound level balanced."

        elif "wrap up" in mode or "clean" in mode or "close all" in mode:
            pyautogui.hotkey("win", "d")
            return "Desktop cleared and all active windows minimized, sir."

        return f"Preset '{mode_name}' is not currently configured in my protocols, sir."

    @staticmethod
    def launch_app(app_target: str) -> str:
        """Launches desktop software, web browsers, or built-in utilities."""
        app = app_target.lower().strip()
        system = platform.system().lower()

        # Windows-specific binary mappings
        app_map_win = {
            "vscode": "code",
            "vs code": "code",
            "code": "code",
            "chrome": "chrome",
            "google chrome": "chrome",
            "brave": "brave",
            "edge": "msedge",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "terminal": "wt.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "explorer": "explorer.exe",
            "files": "explorer.exe",
            "task manager": "taskmgr.exe",
            "spotify": "spotify",
            "discord": "discord",
            "slack": "slack"
        }

        try:
            if system == "windows":
                target_exec = app_map_win.get(app, app)
                if shutil.which(target_exec) or app in app_map_win:
                    subprocess.Popen(target_exec, shell=True)
                else:
                    os.system(f"start {target_exec}")
                return f"Launching {app.capitalize()} immediately, sir."
            else:
                subprocess.Popen([app])
                return f"Opening {app}, sir."
        except Exception as e:
            return f"I encountered an issue opening {app}: {str(e)}"

    @staticmethod
    def open_site(target: str) -> str:
        """Opens common web destinations or queries a search engine."""
        target = target.lower().strip()
        urls = {
            "youtube": "https://youtube.com",
            "github": "https://github.com",
            "linkedin": "https://linkedin.com",
            "chatgpt": "https://chatgpt.com",
            "google": "https://google.com",
            "stackoverflow": "https://stackoverflow.com",
            "reddit": "https://reddit.com"
        }
        
        if target in urls:
            url = urls[target]
        elif target.startswith("http://") or target.startswith("https://"):
            url = target
        elif "." in target and " " not in target:
            url = f"https://{target}"
        else:
            url = f"https://www.google.com/search?q={target.replace(' ', '+')}"

        webbrowser.open(url)
        return f"Navigating to {target} right away, sir."

    @staticmethod
    def search_web(query: str) -> str:
        """Searches Google for the requested query and opens the results."""
        clean_q = query.strip()
        url = f"https://www.google.com/search?q={clean_q.replace(' ', '+')}"
        webbrowser.open(url)
        return f"I have run a search for '{clean_q}', sir."

    @staticmethod
    def media_control(action: str) -> str:
        """Controls system sound and media playback."""
        action = action.lower().strip()
        if "mute" in action:
            pyautogui.press("volumemute")
            return "Audio mute toggled, sir."
        elif "up" in action or "increase" in action or "bada" in action:
            for _ in range(5):
                pyautogui.press("volumeup")
            return "Volume increased, sir."
        elif "down" in action or "decrease" in action or "kam" in action:
            for _ in range(5):
                pyautogui.press("volumedown")
            return "Volume lowered, sir."
        elif "pause" in action or "play" in action:
            pyautogui.press("playpause")
            return "Media playback toggled, sir."
        elif "next" in action:
            pyautogui.press("nexttrack")
            return "Skipping to the next track, sir."
        elif "prev" in action or "previous" in action:
            pyautogui.press("prevtrack")
            return "Playing the previous track, sir."
        return "Unknown audio command, sir."

    @staticmethod
    def window_management(action: str) -> str:
        """Minimizes, closes, switches, or locks active windows."""
        action = action.lower().strip()
        system = platform.system().lower()

        if "close" in action or "band" in action:
            pyautogui.hotkey("alt", "f4")
            return "Active window closed, sir."
        elif "minimize" in action or "chhota" in action:
            pyautogui.hotkey("win", "down")
            return "Window minimized, sir."
        elif "maximize" in action or "bada" in action:
            pyautogui.hotkey("win", "up")
            return "Window maximized, sir."
        elif "switch" in action or "tab" in action:
            pyautogui.hotkey("alt", "tab")
            return "Window context switched, sir."
        elif "desktop" in action:
            pyautogui.hotkey("win", "d")
            return "Showing desktop, sir."
        elif "lock" in action:
            if system == "windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
                return "Workstation locked, sir."
            return "Workstation lock is unsupported on this platform."
        return "Window operation not recognized, sir."

    @staticmethod
    def system_status() -> str:
        """Reports real-time CPU, RAM, Disk, and Battery diagnostics."""
        cpu = psutil.cpu_percent(interval=0.3)
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/" if platform.system().lower() != "windows" else "C:\\").percent
        
        battery = psutil.sensors_battery()
        batt_str = ""
        if battery:
            plugged = "charging" if battery.power_plugged else "on battery power"
            batt_str = f", battery is at {battery.percent}% and {plugged}"

        return f"CPU utilization is at {cpu}%, RAM is at {ram}%, and the primary disk is at {disk}% capacity{batt_str}, sir."

    @staticmethod
    def take_screenshot(filename: str = "screenshot.png") -> str:
        """Captures a screenshot of the main screen and saves it locally."""
        try:
            save_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
            pyautogui.screenshot(save_path)
            return f"Screenshot captured and stored on your Desktop as {filename}, sir."
        except Exception as e:
            return f"Screenshot capture error: {str(e)}"

    @staticmethod
    def run_command(command: str) -> str:
        """Runs a safe shell/terminal command and returns concise stdout."""
        dangerous = ["rmdir /s", "del /f", "format", "mkfs", "rm -rf"]
        if any(d in command.lower() for d in dangerous):
            return "Security alert: That command is restricted by safety protocols, sir."

        try:
            res = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=6
            )
            output = res.stdout.strip() or res.stderr.strip()
            if not output:
                return "Command executed successfully with zero exit errors, sir."
            first_line = output.splitlines()[0][:100]
            return f"Command returned: {first_line}, sir."
        except subprocess.TimeoutExpired:
            return "Command execution timed out after six seconds, sir."
        except Exception as e:
            return f"Execution failed: {str(e)}"
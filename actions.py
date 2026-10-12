import os
import subprocess
import webbrowser
import platform
import shutil
import time
import socket
from typing import Optional, Dict

import pyautogui
import psutil

try:
    import winreg
except ImportError:
    winreg = None

try:
    import win32api
except ImportError:
    win32api = None

from jarvis_memory import JarvisMemory

# Jarvis only executes discrete keyboard shortcuts (volume, media, window keys).
# Disabling FAILSAFE ensures keyboard shortcuts succeed even if the mouse is parked at (0, 0).
pyautogui.FAILSAFE = False

# ----------------- Network Connectivity Helper -----------------
_last_network_check = 0.0
_cached_network_status: Optional[bool] = None


def is_network_available(force_refresh: bool = False, timeout: float = 0.8) -> bool:
    """
    Fast, non-blocking check to verify if external internet connectivity is present.
    Caches the result for 3 seconds to avoid redundant socket queries during rapid calls.
    """
    global _last_network_check, _cached_network_status
    now = time.time()
    if not force_refresh and _cached_network_status is not None and (now - _last_network_check) < 3.0:
        return _cached_network_status

    status = False
    # Check well-known public DNS resolvers on TCP port 53 with short timeout
    for host in ("1.1.1.1", "8.8.8.8"):
        try:
            sock = socket.create_connection((host, 53), timeout=timeout)
            sock.close()
            status = True
            break
        except (socket.timeout, socket.error, OSError):
            continue

    _last_network_check = now
    _cached_network_status = status
    return status


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
        try:
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
                    lines = [l.strip() for l in res.stdout.strip().splitlines() if l.strip()]
                    if lines:
                        parts = lines[0].split(",")
                        if len(parts) >= 2:
                            gpu_util = parts[0].strip()
                            gpu_temp = parts[1].strip()
                            gpu_info = f", GPU at {gpu_util}% and {gpu_temp} degrees Celsius"
            except Exception:
                pass

            return f"Good day, sir. The time is {now_str}. CPU load is at {cpu}%, RAM utilization is at {ram}%, {batt_info}{gpu_info}. All core systems are nominal."
        except Exception as e:
            return f"System briefing unavailable due to sensor error: {str(e)}, sir."

    @staticmethod
    def resolve_app_path(app_target: str) -> Optional[str]:
        """
        Locates the executable path for a requested application on the host machine.
        Checks:
        1. Exact and aliased names via shutil.which
        2. Windows App Paths registry entries (HKLM / HKCU)
        3. Common Windows directory locations
        """
        app = app_target.lower().strip()
        system = platform.system().lower()
        if system != "windows":
            return shutil.which(app)

        # Standard Windows binary and alias mapping
        app_candidates: Dict[str, list] = {
            "vscode": ["code.cmd", "Code.exe", "code"],
            "vs code": ["code.cmd", "Code.exe", "code"],
            "code": ["code.cmd", "Code.exe", "code"],
            "chrome": ["chrome.exe", "chrome"],
            "google chrome": ["chrome.exe", "chrome"],
            "brave": ["brave.exe", "brave"],
            "edge": ["msedge.exe", "msedge"],
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "calc": ["calc.exe"],
            "terminal": ["wt.exe", "powershell.exe", "cmd.exe"],
            "wt": ["wt.exe"],
            "cmd": ["cmd.exe"],
            "command prompt": ["cmd.exe"],
            "powershell": ["powershell.exe"],
            "explorer": ["explorer.exe"],
            "files": ["explorer.exe"],
            "file explorer": ["explorer.exe"],
            "task manager": ["taskmgr.exe"],
            "taskmgr": ["taskmgr.exe"],
            "spotify": ["spotify.exe", "spotify"],
            "discord": ["discord.exe", "discord"],
            "slack": ["slack.exe", "slack"]
        }

        candidates = app_candidates.get(app, [app, f"{app}.exe"])

        # 1. Search PATH via shutil.which
        for c in candidates:
            w = shutil.which(c)
            if w and os.path.exists(w):
                return w

        # 2. Check Windows App Paths in registry
        if winreg:
            for c in candidates:
                reg_name = c if c.lower().endswith(".exe") else f"{c}.exe"
                for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                    try:
                        k = winreg.OpenKey(root, rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{reg_name}")
                        val, _ = winreg.QueryValueEx(k, "")
                        winreg.CloseKey(k)
                        if val and os.path.exists(val):
                            return val
                    except Exception:
                        pass

        # 3. Check well-known installation directories
        common_search_paths = [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Microsoft VS Code\Code.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\wt.exe"),
        ]
        for p in common_search_paths:
            if os.path.exists(p):
                p_lower = os.path.basename(p).lower()
                for c in candidates:
                    if c.lower() in p_lower:
                        return p

        return None

    @staticmethod
    def launch_app(app_target: str) -> str:
        """Launches desktop software, web browsers, or built-in utilities with verification."""
        app = app_target.lower().strip()
        system = platform.system().lower()

        try:
            if system == "windows":
                resolved = DesktopController.resolve_app_path(app)
                if not resolved:
                    return f"I could not locate '{app}' on this system, sir. Please verify that it is installed and accessible."

                use_shell = resolved.lower().endswith((".cmd", ".bat"))
                proc = subprocess.Popen([resolved] if not use_shell else resolved, shell=use_shell)
                time.sleep(0.15)
                # Verify that the process didn't immediately exit with a failure code
                if proc.poll() is not None and proc.returncode != 0:
                    return f"Failed to launch {app.capitalize()}: process exited with error code {proc.returncode}, sir."

                return f"Launching {app.capitalize()} immediately, sir."
            else:
                bin_path = shutil.which(app)
                if not bin_path:
                    return f"I could not locate '{app}' in system PATH, sir."
                subprocess.Popen([bin_path])
                return f"Opening {app}, sir."
        except Exception as e:
            return f"I encountered an issue opening {app}: {str(e)}"

    @staticmethod
    def run_macro(mode_name: str) -> str:
        """Executes combined setup routines for specific work modes, respecting offline state."""
        mode = mode_name.lower().strip()

        if "code" in mode or "dev" in mode or "coding" in mode:
            launched = []
            failed = []

            # 1. VS Code
            res_code = DesktopController.launch_app("vscode")
            if "launching" in res_code.lower() or "opening" in res_code.lower():
                launched.append("Visual Studio Code")
            else:
                failed.append("Visual Studio Code")

            # 2. Terminal
            res_term = DesktopController.launch_app("terminal")
            if "launching" in res_term.lower() or "opening" in res_term.lower():
                launched.append("Terminal")
            else:
                failed.append("Terminal")

            # 3. GitHub (only if online)
            if is_network_available():
                try:
                    webbrowser.open("https://github.com")
                    launched.append("GitHub")
                    note = ""
                except Exception:
                    note = " (GitHub browser launch encountered an issue)."
            else:
                note = " Note that GitHub was omitted because the system is offline."

            if not launched:
                return f"Unable to initiate coding protocol, sir. Could not launch: {', '.join(failed)}."

            return f"Coding protocol initiated, sir. {', '.join(launched)} are active.{note}"

        elif "chill" in mode or "break" in mode or "music" in mode:
            # Balance sound volume locally
            try:
                for _ in range(3):
                    pyautogui.press("volumedown")
            except Exception:
                pass

            if is_network_available():
                webbrowser.open("https://music.youtube.com")
                return "Break mode engaged, sir. Music playback opened and sound level balanced."
            else:
                spot_path = DesktopController.resolve_app_path("spotify")
                if spot_path:
                    DesktopController.launch_app("spotify")
                    return "Break mode engaged, sir. Local Spotify launched and sound level balanced."
                return "Break mode engaged, sir. Sound level balanced. Note that online music streaming is unavailable offline."

        elif "wrap up" in mode or "clean" in mode or "close all" in mode:
            try:
                pyautogui.hotkey("win", "d")
                return "Desktop cleared and all active windows minimized, sir."
            except Exception as e:
                return f"Failed to clear desktop: {str(e)}, sir."

        return f"Preset '{mode_name}' is not currently configured in my protocols, sir."

    @staticmethod
    def open_site(target: str) -> str:
        """Opens common web destinations or queries a search engine, verifying network connectivity."""
        target_clean = target.lower().strip()
        if not target_clean:
            return "Please specify a destination, sir."

        # Detect local files or addresses which can always open offline
        is_local = (
            target_clean.startswith(("localhost", "127.0.0.1", "file://", "http://localhost", "http://127.0.0.1"))
            or target_clean.endswith((".html", ".htm"))
            or os.path.exists(target_clean)
        )

        urls = {
            "youtube": "https://youtube.com",
            "github": "https://github.com",
            "linkedin": "https://linkedin.com",
            "chatgpt": "https://chatgpt.com",
            "google": "https://google.com",
            "stackoverflow": "https://stackoverflow.com",
            "reddit": "https://reddit.com"
        }

        if target_clean in urls:
            url = urls[target_clean]
        elif target_clean.startswith("http://") or target_clean.startswith("https://"):
            url = target_clean
        elif "." in target_clean and " " not in target_clean:
            url = f"https://{target_clean}"
        elif is_local:
            url = target_clean
        else:
            url = f"https://www.google.com/search?q={target_clean.replace(' ', '+')}"

        if is_local:
            try:
                webbrowser.open(url)
                return f"Navigating to local address {target_clean}, sir."
            except Exception as e:
                return f"Failed to navigate to {target_clean}: {str(e)}, sir."

        if not is_network_available():
            return f"Navigating to '{target_clean}' requires an active internet connection, sir. Local offline tools remain ready."

        try:
            webbrowser.open(url)
            return f"Navigating to {target_clean} right away, sir."
        except Exception as e:
            return f"Failed to navigate to {target_clean}: {str(e)}, sir."

    @staticmethod
    def search_web(query: str) -> str:
        """Searches Google for the requested query and opens results when online."""
        clean_q = query.strip()
        if not clean_q:
            return "Please specify a query to search, sir."

        if not is_network_available():
            return f"Web search for '{clean_q}' requires an active internet connection, sir. The system is currently offline."

        try:
            url = f"https://www.google.com/search?q={clean_q.replace(' ', '+')}"
            webbrowser.open(url)
            return f"I have run a search for '{clean_q}', sir."
        except Exception as e:
            return f"Failed to open web search: {str(e)}, sir."

    @staticmethod
    def media_control(action: str) -> str:
        """Controls system sound and media playback."""
        act = action.lower().strip()
        try:
            if "mute" in act or "unmute" in act:
                pyautogui.press("volumemute")
                return "Audio mute toggled, sir."
            elif any(w in act for w in ["up", "increase", "raise", "louder", "bada"]):
                for _ in range(5):
                    pyautogui.press("volumeup")
                return "Volume increased, sir."
            elif any(w in act for w in ["down", "decrease", "lower", "quieter", "kam"]):
                for _ in range(5):
                    pyautogui.press("volumedown")
                return "Volume lowered, sir."
            elif any(w in act for w in ["pause", "play", "resume"]):
                pyautogui.press("playpause")
                return "Media playback toggled, sir."
            elif "stop" in act:
                pyautogui.press("stop")
                return "Media playback stopped, sir."
            elif "next" in act or "skip" in act:
                pyautogui.press("nexttrack")
                return "Skipping to the next track, sir."
            elif any(w in act for w in ["prev", "previous", "back"]):
                pyautogui.press("prevtrack")
                return "Playing the previous track, sir."
            return f"Audio command '{action}' not recognized. Supported controls: mute, volume up/down, play/pause, next, previous."
        except Exception as e:
            return f"Media control error: {str(e)}, sir."

    @staticmethod
    def window_management(action: str) -> str:
        """Minimizes, closes, switches, or locks active windows."""
        act = action.lower().strip()
        system = platform.system().lower()

        try:
            if any(w in act for w in ["close", "exit", "terminate", "band"]):
                pyautogui.hotkey("alt", "f4")
                return "Active window closed, sir."
            elif any(w in act for w in ["minimize", "minimise", "chhota"]):
                pyautogui.hotkey("win", "down")
                return "Window minimized, sir."
            elif any(w in act for w in ["maximize", "maximise", "expand", "bada"]):
                pyautogui.hotkey("win", "up")
                return "Window maximized, sir."
            elif any(w in act for w in ["switch", "tab", "cycle"]):
                pyautogui.hotkey("alt", "tab")
                return "Window context switched, sir."
            elif any(w in act for w in ["desktop", "clear desktop", "hide all"]):
                pyautogui.hotkey("win", "d")
                return "Showing desktop, sir."
            elif "lock" in act:
                if system == "windows":
                    res = subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], timeout=5)
                    if res.returncode == 0:
                        return "Workstation locked, sir."
                    return f"Failed to lock workstation: exit code {res.returncode}, sir."
                return "Workstation lock is unsupported on this platform."
            return f"Window operation '{action}' not recognized. Supported options: close, minimize, maximize, switch, desktop, lock."
        except Exception as e:
            return f"Window management error: {str(e)}, sir."

    @staticmethod
    def system_status() -> str:
        """Reports real-time CPU, RAM, Disk, and Battery diagnostics."""
        try:
            cpu = psutil.cpu_percent(interval=0.2)
            ram = psutil.virtual_memory().percent
            disk_path = "C:\\" if platform.system().lower() == "windows" else "/"
            try:
                disk = psutil.disk_usage(disk_path).percent
                disk_str = f", primary disk is at {disk}% capacity"
            except Exception:
                disk_str = ""

            battery = psutil.sensors_battery()
            batt_str = ""
            if battery:
                plugged = "charging" if battery.power_plugged else "on battery power"
                batt_str = f", battery is at {battery.percent}% and {plugged}"

            return f"CPU utilization is at {cpu}%, RAM is at {ram}%{disk_str}{batt_str}, sir."
        except Exception as e:
            return f"Diagnostics sensor error: {str(e)}, sir."

    @staticmethod
    def take_screenshot(filename: str = "screenshot.png") -> str:
        """Captures a screenshot locally, verifying file creation and image validity."""
        clean_name = os.path.basename(filename.strip()) or "screenshot.png"
        if not clean_name.lower().endswith((".png", ".jpg", ".jpeg")):
            clean_name += ".png"

        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        os.makedirs(desktop_dir, exist_ok=True)
        save_path = os.path.join(desktop_dir, clean_name)

        # 1. Primary method: PowerShell System.Drawing (works in Windows session contexts)
        escaped_path = save_path.replace("'", "''")
        ps_script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
            "$bmp = New-Object System.Drawing.Bitmap $screen.Width, $screen.Height;"
            "$g = [System.Drawing.Graphics]::FromImage($bmp);"
            "$g.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size);"
            f"$bmp.Save('{escaped_path}');"
            "$g.Dispose(); $bmp.Dispose();"
            "Write-Output 'OK'"
        )

        ps_success = False
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True, text=True, timeout=8
            )
            if "OK" in result.stdout and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                ps_success = True
        except Exception:
            ps_success = False

        if ps_success:
            return f"Screenshot captured and saved to your Desktop as {clean_name}, sir."

        # 2. Secondary fallback method: PyAutoGUI
        try:
            pyautogui.screenshot(save_path)
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                return f"Screenshot captured and saved to your Desktop as {clean_name}, sir."
        except Exception as pe:
            return f"Screenshot capture failed: {str(pe)}, sir."

        return f"Screenshot capture failed to produce a valid image at {save_path}, sir."

    @staticmethod
    def run_command(command: str) -> str:
        """Runs a safe shell/terminal command and returns concise stdout, accurately reporting errors."""
        cmd_strip = command.strip()
        if not cmd_strip:
            return "No command was provided to execute, sir."

        dangerous = ["rmdir /s", "del /f", "format", "mkfs", "rm -rf", ":(){ :|:& };:"]
        if any(d in cmd_strip.lower() for d in dangerous):
            return "Security alert: That command is restricted by safety protocols, sir."

        try:
            res = subprocess.run(
                cmd_strip, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=6
            )
            output = res.stdout.strip() or res.stderr.strip()
            if res.returncode != 0:
                first_err = output.splitlines()[0][:100] if output else f"Exit code {res.returncode}"
                return f"Command execution failed with exit code {res.returncode}: {first_err}, sir."

            if not output:
                return "Command executed successfully with zero exit errors, sir."
            first_line = output.splitlines()[0][:100]
            return f"Command returned: {first_line}, sir."
        except subprocess.TimeoutExpired:
            return "Command execution timed out after six seconds, sir."
        except Exception as e:
            return f"Execution failed: {str(e)}"

    @staticmethod
    def get_time_date() -> str:
        """Returns the current formatted date, day, and time."""
        now = time.localtime()
        time_str = time.strftime("%I:%M %p", now)
        date_str = time.strftime("%A, %B %d, %Y", now)
        return f"The time is {time_str} on {date_str}, sir."

    @staticmethod
    def play_youtube(query: str) -> str:
        """Plays or searches for a video or music track on YouTube when online."""
        clean_q = query.strip()
        if not clean_q:
            return "Please specify a song or topic to play on YouTube, sir."

        if not is_network_available():
            return f"Playing '{clean_q}' on YouTube requires an active internet connection, sir. The system is currently offline, but local media and desktop controls remain available."

        try:
            url = f"https://www.youtube.com/results?search_query={clean_q.replace(' ', '+')}"
            webbrowser.open(url)
            return f"Playing '{clean_q}' on YouTube now, sir."
        except Exception as e:
            return f"Failed to open YouTube: {str(e)}, sir."

    @staticmethod
    def get_weather(city: str = "") -> str:
        """Retrieves real-time weather using Open-Meteo API when connected to the internet."""
        if not is_network_available():
            return "Live atmospheric and weather reports require an active internet connection, sir. The system is currently offline, so live weather data cannot be fetched."

        import requests
        target_city = city.strip() if city and city.strip() else "auto"
        try:
            lat, lon, place_name = None, None, ""
            if target_city == "auto":
                try:
                    ip_res = requests.get("https://ipapi.co/json/", timeout=2.5).json()
                    lat = ip_res.get("latitude")
                    lon = ip_res.get("longitude")
                    place_name = ip_res.get("city", "your location")
                except Exception:
                    lat, lon, place_name = 28.6139, 77.2090, "New Delhi"
            else:
                geo_res = requests.get(
                    f"https://geocoding-api.open-meteo.com/v1/search?name={target_city}&count=1", 
                    timeout=2.5
                ).json()
                results = geo_res.get("results")
                if results:
                    lat = results[0].get("latitude")
                    lon = results[0].get("longitude")
                    place_name = results[0].get("name", target_city)
                else:
                    return f"I could not locate geographical coordinates for {target_city}, sir."

            weather_res = requests.get(
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true",
                timeout=2.5
            ).json()
            curr = weather_res.get("current_weather", {})
            temp = curr.get("temperature", "N/A")
            wind = curr.get("windspeed", "N/A")
            code = curr.get("weathercode", 0)

            conditions = {
                0: "clear skies", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "rime fog", 51: "light drizzle", 61: "slight rain",
                63: "moderate rain", 65: "heavy rain", 71: "light snow", 80: "rain showers",
                95: "thunderstorms"
            }
            condition_desc = conditions.get(code, "fair")
            return f"Current weather in {place_name} is {temp} degrees Celsius with {condition_desc} and wind speeds of {wind} kilometers per hour, sir."
        except Exception as e:
            return f"I was unable to retrieve atmospheric data at this moment, sir: {str(e)}"
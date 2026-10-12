import tkinter as tk
import threading
import queue

class StatusPill:
    def __init__(self):
        self.cmd_queue = queue.Queue()
        self.root = None
        self.canvas = None
        self._drag_data = {"x": 0, "y": 0}

        # Color palette for states (Dark theme)
        self.THEMES = {
            "IDLE": {
                "dot": "#6b7280",       # Slate gray
                "text": "J.A.R.V.I.S.: Idle",
                "sub": "Say 'Hey Jarvis'",
                "border": "#27272a"
            },
            "LISTENING": {
                "dot": "#22c55e",       # Vivid emerald green
                "text": "J.A.R.V.I.S.: Listening...",
                "sub": "Speak now...",
                "border": "#15803d"
            },
            "VERIFYING": {
                "dot": "#eab308",       # Warm Amber
                "text": "J.A.R.V.I.S.: Verifying Voice...",
                "sub": "Authenticating voice biometrics",
                "border": "#a16207"
            },
            "THINKING": {
                "dot": "#eab308",       # Warm Amber
                "text": "J.A.R.V.I.S.: Thinking...",
                "sub": "Processing action...",
                "border": "#a16207"
            },
            "SPEAKING": {
                "dot": "#38bdf8",       # Neon Cyan
                "text": "J.A.R.V.I.S.: Speaking",
                "sub": "Playing response",
                "border": "#0284c7"
            },
            "REJECTED": {
                "dot": "#ef4444",       # Alert Red
                "text": "J.A.R.V.I.S.: Unverified Voice",
                "sub": "Voice authentication failed",
                "border": "#991b1b"
            }
        }

    def _draw_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        """Draws a smooth rounded rectangle on the Tkinter canvas."""
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1,
            x2, y1 + r,
            x2, y2 - r,
            x2, y2,
            x2 - r, y2,
            x1 + r, y2,
            x1, y2,
            x1, y2 - r,
            x1, y1 + r,
            x1, y1
        ]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    def _render_pill(self, state_name: str):
        state = self.THEMES.get(state_name, self.THEMES["IDLE"])
        self.canvas.delete("all")

        # Outer rounded pill (Chroma dark background #121214)
        self._draw_rounded_rect(
            2, 2, 258, 54, 26,
            fill="#121214",
            outline=state["border"],
            width=2
        )

        # Status indicator glowing dot
        self.canvas.create_oval(18, 22, 32, 36, fill=state["dot"], outline="")

        # Primary label
        self.canvas.create_text(
            44, 20,
            text=state["text"],
            fill="#f4f4f5",
            font=("Segoe UI", 10, "bold"),
            anchor="w"
        )

        # Subtitle / Instruction label
        self.canvas.create_text(
            44, 38,
            text=state["sub"],
            fill="#a1a1aa",
            font=("Segoe UI", 8),
            anchor="w"
        )

    def _start_drag(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def _on_drag(self, event):
        deltax = event.x - self._drag_data["x"]
        deltay = event.y - self._drag_data["y"]
        new_x = self.root.winfo_x() + deltax
        new_y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{new_x}+{new_y}")

    def _process_queue(self):
        """Processes state change signals safely on the Tkinter UI thread."""
        try:
            while not self.cmd_queue.empty():
                state_name = self.cmd_queue.get_nowait()
                self._render_pill(state_name)
        except queue.Empty:
            pass
        self.root.after(50, self._process_queue)

    def set_state(self, state_name: str):
        """Call this from your background assistant loop (thread-safe)."""
        self.cmd_queue.put(state_name)

    def run(self):
        """Starts the Tkinter event loop."""
        self.root = tk.Tk()
        self.root.overrideredirect(True)      # Remove OS titlebar/borders
        self.root.wm_attributes("-topmost", True)  # Always stay on top

        # Windows background transparency key
        self.root.wm_attributes("-transparentcolor", "#010101")
        self.root.config(bg="#010101")

        # Position at top right of the primary display
        screen_w = self.root.winfo_screenwidth()
        pill_w, pill_h = 262, 58
        start_x = screen_w - pill_w - 40
        start_y = 40
        self.root.geometry(f"{pill_w}x{pill_h}+{start_x}+{start_y}")

        self.canvas = tk.Canvas(
            self.root,
            width=pill_w,
            height=pill_h,
            bg="#010101",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Enable click & drag anywhere on the pill
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Initial render
        self._render_pill("IDLE")

        # Start background signal polling loop
        self.root.after(50, self._process_queue)
        self.root.mainloop()


def start_overlay() -> StatusPill:
    """Helper that spawns the overlay in its own daemon thread."""
    pill = StatusPill()
    t = threading.Thread(target=pill.run, daemon=True)
    t.start()
    return pill


if __name__ == "__main__":
    # Test script demonstrating transitions
    import time
    ui = start_overlay()
    time.sleep(2)
    ui.set_state("LISTENING")
    time.sleep(2)
    ui.set_state("THINKING")
    time.sleep(2)
    ui.set_state("SPEAKING")
    time.sleep(2)
    ui.set_state("IDLE")
    # Keep alive for demo
    while True:
        time.sleep(1)
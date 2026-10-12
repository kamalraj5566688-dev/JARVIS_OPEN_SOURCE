import json
import os
import time
from typing import Dict, Any, List, Optional

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "jarvis_memory.json")

DEFAULT_MEMORY = {
    "user_profile": {
        "name": "Sir",
        "role": "Lead Engineer / Creator",
        "default_browser": "chrome",
        "preferred_editor": "vscode",
        "music_platform": "youtube_music",
        "notes": "Authorized owner with verified voice biometric profile."
    },
    "preferences": {
        "voice_style": "British, formal, concise, calm",
        "response_brevity": "short",  # 1-2 sentences for spoken audio
        "safe_mode": True
    },
    "facts": {
        "workspace": "E:/VOICE ASS",
        "assistant_name": "J.A.R.V.I.S."
    },
    "recent_interactions": []
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges `override` into `base`, preserving nested default keys."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class JarvisMemory:
    """
    OpenJarvis Persistent Memory Layer.
    Maintains long-term user profile, persistent facts, and conversation history
    across system restarts without catastrophic forgetting.
    """
    def __init__(self, memory_path: str = MEMORY_FILE):
        self.memory_path = memory_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    return _deep_merge(DEFAULT_MEMORY, loaded)
            except Exception as e:
                print(f"[Memory] Warning: Could not read memory file ({e}). Initializing defaults.")
        return DEFAULT_MEMORY.copy()

    def save(self) -> bool:
        """Atomically saves memory data to disk and verifies write."""
        tmp_path = f"{self.memory_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.memory_path)
            return True
        except Exception as e:
            print(f"[Memory] Error saving persistent memory: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return False

    # ---------------- Fact & Preference Management ----------------

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.data.get("facts", {}).get(key.lower(), default)

    def set_fact(self, key: str, value: Any) -> str:
        """Stores or updates a long-term fact with persistence verification."""
        clean_key = key.strip().lower()
        if not clean_key:
            return "Fact key cannot be empty, sir."
        if "facts" not in self.data:
            self.data["facts"] = {}
        self.data["facts"][clean_key] = value
        saved = self.save()
        if saved and self.get_fact(clean_key) == value:
            return f"I have committed '{key}' to permanent memory, sir."
        return f"Failed to commit '{key}' to permanent memory due to a storage failure, sir."

    def get_user_name(self) -> str:
        return self.data.get("user_profile", {}).get("name", "Sir")

    def set_user_name(self, name: str) -> bool:
        if "user_profile" not in self.data:
            self.data["user_profile"] = {}
        self.data["user_profile"]["name"] = name.strip()
        return self.save()

    # ---------------- Interaction Logging ----------------

    def record_turn(self, user_text: str, assistant_text: str, tool_used: Optional[str] = None):
        """Records the latest turn into short-term episodic memory (keeps last 20)."""
        turn = {
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_text,
            "assistant": assistant_text,
            "tool": tool_used
        }
        history: List[Dict[str, Any]] = self.data.setdefault("recent_interactions", [])
        history.append(turn)
        if len(history) > 20:
            self.data["recent_interactions"] = history[-20:]
        self.save()

    def get_recent_history(self, limit: int = 4) -> List[Dict[str, str]]:
        """Returns message list formatted for LLM context injection."""
        history = self.data.get("recent_interactions", [])[-limit:]
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        return messages

    # ---------------- Prompt Context Assembly ----------------

    def build_system_context(self) -> str:
        """
        Constructs dynamic memory context for injection into Jarvis's system prompt.
        """
        user_name = self.get_user_name()
        facts = self.data.get("facts", {})
        facts_summary = ", ".join([f"{k}: {v}" for k, v in list(facts.items())[:6]]) if facts else "None"
        
        context_str = (
            f"\n[JARVIS MEMORY CONTEXT]\n"
            f"- Addressing User As: {user_name}\n"
            f"- Known Facts: {facts_summary}\n"
            f"- System Time: {time.strftime('%I:%M %p, %A')}\n"
            f"- Workspace: {facts.get('workspace', 'Active desktop')}\n"
        )
        return context_str

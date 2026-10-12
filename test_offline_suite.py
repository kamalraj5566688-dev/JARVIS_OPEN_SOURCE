"""
J.A.R.V.I.S. Offline Capabilities and System Robustness Test Suite
===================================================================
Comprehensive verification covering:
1. Successful offline execution of core commands
2. Missing dependencies & unavailable resources (no false positives)
3. Offline handling of network-dependent operations
4. Long-running operation timeouts and recovery behavior
5. Cognitive reflex & heuristic fallback when local neural engine is offline
6. Persistence verification and error resilience in JarvisMemory
7. Response sanitization for clean Text-To-Speech output
"""

import os
import sys
import unittest
import time
from unittest.mock import patch, MagicMock

# Ensure current directory is in sys.path
_base_dir = os.path.dirname(os.path.abspath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

import actions
from actions import DesktopController, is_network_available
from jarvis_memory import JarvisMemory, DEFAULT_MEMORY
from jarvis_brain import (
    JarvisBrain,
    normalize_voice_transcript,
    sanitize_for_speech,
)


class TestOfflineCoreCommands(unittest.TestCase):
    """Verifies that all supported offline commands work using local resources."""

    def test_get_time_date(self):
        """Date and time command should execute 100% locally."""
        res = DesktopController.get_time_date()
        self.assertIsInstance(res, str)
        self.assertIn("The time is", res)
        self.assertIn("sir.", res)

    def test_system_status(self):
        """Hardware telemetry should report CPU, RAM, and Disk without network access."""
        res = DesktopController.system_status()
        self.assertIsInstance(res, str)
        self.assertIn("CPU utilization is at", res)
        self.assertIn("RAM is at", res)
        self.assertIn("sir.", res)

    def test_system_briefing(self):
        """System briefing provides time, CPU, RAM, and power state locally."""
        res = DesktopController.system_briefing()
        self.assertIsInstance(res, str)
        self.assertIn("Good day, sir", res)
        self.assertIn("CPU load is at", res)
        self.assertIn("RAM utilization is at", res)

    def test_take_screenshot(self):
        """Screenshots should be captured and verified on disk, returning clear status."""
        test_filename = f"test_offline_screenshot_{int(time.time())}.png"
        desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        expected_path = os.path.join(desktop_dir, test_filename)

        try:
            res = DesktopController.take_screenshot(test_filename)
            self.assertIsInstance(res, str)
            if "Screenshot captured" in res:
                self.assertTrue(os.path.exists(expected_path), "Screenshot file must exist")
                self.assertGreater(os.path.getsize(expected_path), 0, "Screenshot file must not be empty")
            else:
                # If running in non-interactive/headless CI environment, verify error is explicit
                self.assertTrue("failed" in res.lower() or "error" in res.lower())
        finally:
            if os.path.exists(expected_path):
                try:
                    os.remove(expected_path)
                except Exception:
                    pass

    def test_media_control(self):
        """Media controls (mute, volume, playback) should handle actions locally."""
        res_mute = DesktopController.media_control("mute")
        self.assertIn("Audio mute toggled", res_mute)

        res_up = DesktopController.media_control("volume up")
        self.assertIn("Volume increased", res_up)

        res_down = DesktopController.media_control("volume down")
        self.assertIn("Volume lowered", res_down)

        res_pause = DesktopController.media_control("pause")
        self.assertIn("Media playback toggled", res_pause)

        res_next = DesktopController.media_control("next track")
        self.assertIn("Skipping to the next track", res_next)

        res_prev = DesktopController.media_control("prev track")
        self.assertIn("Playing the previous track", res_prev)

        # Unrecognized media action must report supported controls
        res_unk = DesktopController.media_control("explode sound")
        self.assertIn("not recognized", res_unk)
        self.assertIn("Supported controls", res_unk)

    def test_window_management(self):
        """Window management handles supported commands and rejects unknown actions clearly."""
        res_desktop = DesktopController.window_management("desktop")
        self.assertIn("Showing desktop", res_desktop)

        res_min = DesktopController.window_management("minimize")
        self.assertIn("Window minimized", res_min)

        res_max = DesktopController.window_management("maximize")
        self.assertIn("Window maximized", res_max)

        res_sw = DesktopController.window_management("switch")
        self.assertIn("Window context switched", res_sw)

        res_unk = DesktopController.window_management("teleport window")
        self.assertIn("not recognized", res_unk)
        self.assertIn("Supported options", res_unk)

    def test_learning_status(self):
        """Learning status reads local state and trace logs without internet."""
        res = DesktopController.learning_status()
        self.assertIsInstance(res, str)
        self.assertTrue(
            "Autonomous learning is active" in res
            or "evolution cycles" in res
            or "running an autonomous fine-tuning cycle" in res
            or "encountered an issue" in res
        )


class TestAppResolutionAndVerification(unittest.TestCase):
    """Verifies that application launches are accurately located and never falsely reported."""

    def test_resolve_known_system_apps(self):
        """System apps like cmd, powershell, or notepad should resolve to local executables."""
        notepad_path = DesktopController.resolve_app_path("notepad")
        self.assertIsNotNone(notepad_path)
        self.assertTrue(os.path.exists(notepad_path))

        cmd_path = DesktopController.resolve_app_path("cmd")
        self.assertIsNotNone(cmd_path)
        self.assertTrue(os.path.exists(cmd_path))

    def test_nonexistent_app_fails_clearly(self):
        """Nonexistent applications must NOT claim to launch; they must fail clearly."""
        fake_app = "definitely_nonexistent_software_xyz123"
        res = DesktopController.launch_app(fake_app)
        self.assertIn("could not locate", res.lower())
        self.assertIn(fake_app, res.lower())
        self.assertNotIn("launching immediately", res.lower())


class TestCommandExecutionRobustness(unittest.TestCase):
    """Verifies safe terminal execution, error reporting, and timeouts."""

    def test_run_command_success(self):
        """Safe command with exit code 0 returns actual output."""
        res = DesktopController.run_command("echo OfflineCommandVerification")
        self.assertIn("OfflineCommandVerification", res)

    def test_run_command_nonzero_exit_code_fails_clearly(self):
        """Commands that exit with a non-zero exit code must be reported as failures, never success."""
        # 'exit 42' in powershell / cmd returns exit code 42
        res = DesktopController.run_command("cmd /c exit 42")
        self.assertIn("failed", res.lower())
        self.assertIn("42", res)
        self.assertNotIn("zero exit errors", res.lower())

    def test_run_command_timeout(self):
        """Commands that hang or take too long are cleanly terminated with timeout status."""
        # Sleep for 10 seconds, but run_command timeout is 6 seconds
        res = DesktopController.run_command('powershell -Command "Start-Sleep -Seconds 10"')
        self.assertIn("timed out", res.lower())

    def test_run_command_security_restriction(self):
        """Destructive or dangerous commands are rejected by safety protocols."""
        res = DesktopController.run_command("rmdir /s /q test_dir")
        self.assertIn("Security alert", res)
        self.assertIn("restricted by safety protocols", res)


class TestNetworkDependentCommandsOffline(unittest.TestCase):
    """Verifies that online commands fail clearly and explain what is needed when offline."""

    @patch("actions.is_network_available", return_value=False)
    def test_search_web_offline(self, mock_net):
        """Search web when offline must fail clearly without opening broken browser tabs."""
        res = DesktopController.search_web("quantum computing")
        self.assertIn("requires an active internet connection", res)
        self.assertIn("offline", res.lower())

    @patch("actions.is_network_available", return_value=False)
    def test_play_youtube_offline(self, mock_net):
        """YouTube playback when offline must explain that network access is required."""
        res = DesktopController.play_youtube("Iron Man theme")
        self.assertIn("requires an active internet connection", res)
        self.assertIn("offline", res.lower())

    @patch("actions.is_network_available", return_value=False)
    def test_get_weather_offline(self, mock_net):
        """Weather queries when offline must explain that network access is required."""
        res = DesktopController.get_weather("New York")
        self.assertIn("active internet connection", res.lower())
        self.assertIn("offline", res.lower())

    @patch("actions.is_network_available", return_value=False)
    def test_open_site_remote_offline(self, mock_net):
        """Remote websites when offline must not pretend to succeed."""
        res = DesktopController.open_site("youtube")
        self.assertIn("requires an active internet connection", res)

    @patch("actions.is_network_available", return_value=False)
    def test_open_site_local_offline(self, mock_net):
        """Local file or localhost addresses must still be permitted offline."""
        with patch("webbrowser.open") as mock_open:
            res = DesktopController.open_site("http://localhost:8000")
            self.assertIn("Navigating to local address", res)
            mock_open.assert_called_once()

    @patch("actions.is_network_available", return_value=False)
    def test_run_macro_coding_offline(self, mock_net):
        """Coding macro offline opens local tools (VS Code, Terminal) and notes GitHub was omitted."""
        with patch.object(DesktopController, "launch_app", return_value="Launching Terminal immediately, sir."):
            res = DesktopController.run_macro("coding")
            self.assertIn("Coding protocol initiated", res)
            self.assertIn("offline", res.lower())
            self.assertIn("GitHub was omitted", res)

    @patch("actions.is_network_available", return_value=False)
    def test_run_macro_chill_offline(self, mock_net):
        """Break mode offline balances volume and notes online streaming is unavailable."""
        res = DesktopController.run_macro("chill")
        self.assertIn("Break mode engaged", res)
        self.assertIn("sound level balanced", res.lower())
        self.assertIn("offline", res.lower())


class TestMemoryPersistenceAndVerification(unittest.TestCase):
    """Verifies that persistent memory commits data atomically and verifies write."""

    def setUp(self):
        self.test_memory_path = os.path.join(_base_dir, "test_temp_memory.json")
        if os.path.exists(self.test_memory_path):
            os.remove(self.test_memory_path)
        self.memory = JarvisMemory(memory_path=self.test_memory_path)

    def tearDown(self):
        if os.path.exists(self.test_memory_path):
            try:
                os.remove(self.test_memory_path)
            except Exception:
                pass
        if os.path.exists(f"{self.test_memory_path}.tmp"):
            try:
                os.remove(f"{self.test_memory_path}.tmp")
            except Exception:
                pass

    def test_set_and_verify_fact(self):
        """Facts committed must be verified and persisted across instances."""
        res = self.memory.set_fact("preferred_ide", "VS Code Insiders")
        self.assertIn("committed", res.lower())
        self.assertIn("preferred_ide", res)

        # Reload from fresh instance to verify disk persistence
        fresh_memory = JarvisMemory(memory_path=self.test_memory_path)
        self.assertEqual(fresh_memory.get_fact("preferred_ide"), "VS Code Insiders")

    def test_empty_fact_key_rejected(self):
        """Empty keys must be rejected with diagnostic explanation."""
        res = self.memory.set_fact("", "value")
        self.assertIn("cannot be empty", res.lower())

    def test_storage_failure_fails_clearly(self):
        """If saving to disk fails, set_fact must report failure and not pretend to succeed."""
        with patch.object(self.memory, "save", return_value=False):
            res = self.memory.set_fact("test_key", "test_val")
            self.assertIn("failed", res.lower())
            self.assertIn("storage failure", res.lower())


class TestJarvisBrainFastHeuristicFallback(unittest.TestCase):
    """Verifies fast cognitive heuristic fallback covering all commands when Ollama is offline."""

    def setUp(self):
        self.brain = JarvisBrain()

    def test_fallback_time_date(self):
        res = self.brain.fast_heuristic_fallback("what time is it")
        self.assertIsNotNone(res)
        self.assertIn("The time is", res)

    def test_fallback_system_telemetry(self):
        res = self.brain.fast_heuristic_fallback("give me system briefing")
        self.assertIsNotNone(res)
        self.assertIn("CPU load is at", res)

    def test_fallback_window_controls(self):
        res_min = self.brain.fast_heuristic_fallback("minimize window")
        self.assertIn("Window minimized", res_min)

        res_close = self.brain.fast_heuristic_fallback("close window")
        self.assertIn("Active window closed", res_close)

        res_max = self.brain.fast_heuristic_fallback("maximize window")
        self.assertIn("Window maximized", res_max)

        res_sw = self.brain.fast_heuristic_fallback("switch window")
        self.assertIn("Window context switched", res_sw)

    def test_fallback_media_controls(self):
        res = self.brain.fast_heuristic_fallback("mute audio")
        self.assertIn("Audio mute toggled", res)

        res_up = self.brain.fast_heuristic_fallback("volume up")
        self.assertIn("Volume increased", res_up)

    def test_fallback_remember_fact(self):
        res = self.brain.fast_heuristic_fallback("remember that my workspace is in E drive")
        self.assertIsNotNone(res)
        self.assertIn("committed", res.lower())

    def test_fallback_learning_status(self):
        res = self.brain.fast_heuristic_fallback("what is your learning status")
        self.assertIsNotNone(res)
        self.assertTrue("Autonomous learning" in res or "evolution cycles" in res or "encountered an issue" in res)

    def test_fallback_safe_terminal_command(self):
        res = self.brain.fast_heuristic_fallback("run command echo OfflineBrainTest")
        self.assertIsNotNone(res)
        self.assertIn("OfflineBrainTest", res)

    @patch("ollama.Client")
    def test_process_query_recovers_when_ollama_offline(self, mock_client_cls):
        """When Ollama throws connection error, process_query recovers via fast heuristic fallback."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("Could not connect to Ollama daemon on localhost:11434")
        mock_client_cls.return_value = mock_client

        response = self.brain.process_query("what is the current time")
        self.assertIsInstance(response, str)
        self.assertIn("The time is", response)
        self.assertNotIn("Please ensure Ollama is active", response)


class TestVoiceTextSanitization(unittest.TestCase):
    """Verifies transcript normalization and clean vocal sanitization for TTS."""

    def test_normalize_voice_transcript(self):
        raw = "Hey Jarvis, please can you open vs cold"
        cleaned = normalize_voice_transcript(raw)
        self.assertEqual(cleaned, "open vscode")

    def test_sanitize_for_speech_strips_markdown_and_emojis(self):
        raw = "```json\n{'tool': 'test'}\n``` **Sir**, CPU is at *24%*! 🚀 [View details](http://test)"
        sanitized = sanitize_for_speech(raw)
        self.assertNotIn("*", sanitized)
        self.assertNotIn("`", sanitized)
        self.assertNotIn("🚀", sanitized)
        self.assertNotIn("http://", sanitized)
        self.assertTrue(sanitized.endswith("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)

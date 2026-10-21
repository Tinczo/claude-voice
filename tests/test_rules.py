"""Per-session and per-project rules for which sessions speak, in which voice."""

import io
import os
import tempfile
import time
import unittest
from unittest import mock

import claude_voice as cv


class VoiceRuleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        os.makedirs(os.path.join(self.root, "proj", "sub"))
        patcher = mock.patch.object(cv, "SESSIONS_PATH", os.path.join(self.root, "sessions.json"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.cfg = dict(cv.default_config(), projects={
            self.root: {"voice": "am_adam"},
            os.path.join(self.root, "proj"): {"enabled": False, "voice": "bf_emma"},
        })

    def test_default_applies_without_rules(self):
        rule = cv.voice_rule(dict(self.cfg, projects={}), None, self.root)
        self.assertTrue(rule["enabled"])
        self.assertIsNone(rule["voice"])

    def test_longest_project_prefix_wins(self):
        rule = cv.voice_rule(self.cfg, None, os.path.join(self.root, "proj", "sub"))
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["voice"], "bf_emma")

    def test_prefix_does_not_match_sibling_directory_name(self):
        rule = cv.voice_rule(dict(self.cfg, default_enabled=False), None, self.root + "-other")
        self.assertFalse(rule["enabled"])
        self.assertEqual(rule["source"], "default")

    def test_session_override_beats_project(self):
        cv._save_sessions({"s1": {"enabled": True, "voice": "bm_george", "t": time.time()}})
        rule = cv.voice_rule(self.cfg, "s1", os.path.join(self.root, "proj"))
        self.assertTrue(rule["enabled"])
        self.assertEqual(rule["voice"], "bm_george")

    def test_speed_follows_the_same_precedence(self):
        cfg = dict(self.cfg, projects={self.root: {"speed": 1.0}})
        self.assertIsNone(cv.voice_rule(dict(cfg, projects={}), None, self.root)["speed"])
        self.assertEqual(cv.voice_rule(cfg, None, self.root)["speed"], 1.0)
        cv._save_sessions({"s1": {"speed": 1.5, "t": time.time()}})
        self.assertEqual(cv.voice_rule(cfg, "s1", self.root)["speed"], 1.5)

    def test_stale_session_overrides_expire(self):
        cv._save_sessions({"s1": {"enabled": True, "t": time.time() - cv.SESSION_TTL - 1}})
        rule = cv.voice_rule(self.cfg, "s1", os.path.join(self.root, "proj"))
        self.assertFalse(rule["enabled"])


class SpeedRuleCommandTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = os.path.realpath(tmp.name)
        self.cfg = cv.default_config()
        for patcher in (
            mock.patch.object(cv, "RUNTIME_DIR", self.root),
            mock.patch.object(cv, "SESSIONS_PATH", os.path.join(self.root, "sessions.json")),
            mock.patch.object(cv, "load_config", return_value=self.cfg),
            mock.patch.object(cv, "save_config"),
            mock.patch.dict(os.environ, {"CLAUDE_CODE_SESSION_ID": "s1",
                                         "CLAUDE_PROJECT_DIR": self.root}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def slash(self, *args):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            cv.cmd_slash(list(args))
        return out.getvalue()

    def test_project_speed_is_saved_and_shown(self):
        out = self.slash("project", "speed", "1.1x")
        self.assertEqual(self.cfg["projects"][self.root], {"speed": 1.1})
        self.assertIn("1.1x · 100% (from project", out)

    def test_project_volume_accepts_percent_and_zero(self):
        out = self.slash("project", "volume", "70%")
        self.assertEqual(self.cfg["projects"][self.root], {"volume": 0.7})
        self.assertIn("· 70% (from project", out)
        self.slash("project", "volume", "0")
        self.assertEqual(cv.voice_rule(self.cfg, None, self.root)["volume"], 0.0)

    def test_session_speed_overrides_and_clears(self):
        self.slash("session", "speed", "0.9")
        self.assertEqual(cv.voice_rule(self.cfg, "s1", self.root)["speed"], 0.9)
        self.slash("session", "clear")
        self.assertIsNone(cv.voice_rule(self.cfg, "s1", self.root)["speed"])

    def test_bad_speed_is_reported_not_saved(self):
        self.assertIn("error:", self.slash("project", "speed", "fast"))
        cv.save_config.assert_not_called()

    def test_speeds_are_clamped(self):
        self.assertEqual(cv.parse_speed("9"), 4.0)
        self.assertEqual(cv.parse_speed("0.1×"), 0.25)
        with self.assertRaises(ValueError):
            cv.parse_speed("nan")

    def test_volumes_read_as_gain_or_percent_and_are_clamped(self):
        self.assertEqual(cv.parse_volume("0.7"), 0.7)
        self.assertEqual(cv.parse_volume("70%"), 0.7)
        self.assertEqual(cv.parse_volume("500"), 2.0)
        with self.assertRaises(ValueError):
            cv.parse_volume("inf")


if __name__ == "__main__":
    unittest.main()

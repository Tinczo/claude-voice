"""Privacy and behavior tests for opt-in spoken-response history."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import claude_voice as cv


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cv-history-test-")
        self.runtime_dir = Path(self._tmp.name) / "runtime"
        self.history_path = self.runtime_dir / "history.jsonl"
        self._runtime = cv.RUNTIME_DIR
        self._history = cv.HISTORY_PATH
        self._limit = cv.HISTORY_MAX_ENTRIES
        self._config = cv._config
        cv.RUNTIME_DIR = str(self.runtime_dir)
        cv.HISTORY_PATH = str(self.history_path)
        cv._config = None

    def tearDown(self):
        cv.RUNTIME_DIR = self._runtime
        cv.HISTORY_PATH = self._history
        cv.HISTORY_MAX_ENTRIES = self._limit
        cv._config = self._config
        self._tmp.cleanup()

    def test_history_is_opt_in(self):
        self.assertFalse(cv.default_config()["history_enabled"])

    def test_history_file_is_private_and_bounded(self):
        old_umask = os.umask(0o022)
        cv.HISTORY_MAX_ENTRIES = 3
        try:
            for i in range(5):
                cv._log_history(f"message {i}", "system", "default")
        finally:
            os.umask(old_umask)

        if os.name == "posix":  # Windows privacy is an ACL; see test_windows_compat
            file_mode = stat.S_IMODE(self.history_path.stat().st_mode)
            dir_mode = stat.S_IMODE(self.runtime_dir.stat().st_mode)
            self.assertEqual(file_mode, 0o600)
            self.assertEqual(dir_mode, 0o700)
        self.assertEqual(
            [entry["text"] for entry in cv._read_history()],
            ["message 4", "message 3", "message 2"],
        )

    @unittest.skipUnless(os.name == "posix", "Unix permission bits")
    def test_existing_prototype_history_permissions_are_repaired(self):
        self.runtime_dir.mkdir(mode=0o755)
        self.history_path.write_text('{"text":"old entry"}\n', encoding="utf-8")
        os.chmod(self.runtime_dir, 0o755)
        os.chmod(self.history_path, 0o644)

        cv._harden_history_storage()

        self.assertEqual(stat.S_IMODE(self.runtime_dir.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(self.history_path.stat().st_mode), 0o600)

    def test_read_history_skips_malformed_or_non_object_rows(self):
        self.runtime_dir.mkdir(mode=0o700)
        with self.history_path.open("w", encoding="utf-8") as f:
            f.write("not-json\n")
            f.write(json.dumps(["not", "an", "entry"]) + "\n")
            f.write(json.dumps({"text": "kept", "ts": 1}) + "\n")

        self.assertEqual(cv._read_history(), [{"text": "kept", "ts": 1}])

    def test_replay_does_not_create_a_duplicate_history_entry(self):
        sent = []

        def fake_send(payload, timeout=2.0):
            sent.append((payload, timeout))
            return {"queued": True}

        with (
            mock.patch.object(cv, "_read_history", return_value=[{"text": "repeat me"}]),
            mock.patch.object(cv, "_ensure_daemon", return_value=True),
            mock.patch.object(cv, "_resolve_tty", return_value="/dev/null"),
            mock.patch.object(cv, "_send_to_daemon", side_effect=fake_send),
            mock.patch.object(cv, "load_config", return_value={"provider": "system"}),
        ):
            cv.cmd_replay([])

        self.assertEqual(len(sent), 1)
        self.assertFalse(sent[0][0]["record_history"])


if __name__ == "__main__":
    unittest.main()

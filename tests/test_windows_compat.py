"""Windows support: TCP daemon transport, locking, console and setup paths.

Most tests run on every platform: the TCP transport can be forced on, and
the Windows-only branches are pure logic behind the `WINDOWS` flag.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

import claude_voice as cv

REPO_ROOT = Path(__file__).resolve().parent.parent


def _raw_request(port: int, payload: dict) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
        s.sendall(json.dumps(payload).encode() + b"\n")
        return json.loads(s.makefile("rb").readline())


class TcpTransportTests(unittest.TestCase):
    """A daemon forced onto the localhost TCP transport used on Windows."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cv-tcp-test-")
        self.addr_path = os.path.join(self.tmp, "daemon.addr")
        bootstrap = (
            "import sys, types\n"
            "fake_kokoro = types.ModuleType('kokoro')\n"
            "fake_kokoro.KPipeline = lambda *a, **kw: (lambda *a, **kw: [])\n"
            "sys.modules['kokoro'] = fake_kokoro\n"
            "import claude_voice\n"
            "claude_voice.DAEMON_TCP = True\n"
            f"claude_voice.RUNTIME_DIR = {self.tmp!r}\n"
            f"claude_voice.SOCK_PATH = {os.path.join(self.tmp, 'daemon.sock')!r}\n"
            f"claude_voice.ADDR_PATH = {self.addr_path!r}\n"
            f"claude_voice.PID_PATH = {os.path.join(self.tmp, 'daemon.pid')!r}\n"
            f"claude_voice.LOG_PATH = {os.path.join(self.tmp, 'daemon.log')!r}\n"
            f"claude_voice.CONFIG_PATH = {os.path.join(self.tmp, 'config.json')!r}\n"
            "claude_voice.cmd_daemon()\n"
        )
        env = dict(os.environ, PYTHONPATH=f"{REPO_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        self.proc = subprocess.Popen([sys.executable, "-c", bootstrap], env=env,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=str(REPO_ROOT))
        self.patches = [
            mock.patch.object(cv, "DAEMON_TCP", True),
            mock.patch.object(cv, "ADDR_PATH", self.addr_path),
        ]
        for p in self.patches:
            p.start()
        deadline = time.monotonic() + 15
        while not cv._daemon_alive():
            if time.monotonic() > deadline or self.proc.poll() is not None:
                self.proc.kill()
                out, err = self.proc.communicate(timeout=2)
                self.fail(f"daemon never came up.\nstdout: {out!r}\nstderr: {err!r}")
            time.sleep(0.1)
        with open(self.addr_path, encoding="utf-8") as f:
            self.addr = json.load(f)

    def tearDown(self):
        cv._send_to_daemon({"op": "shutdown"}, timeout=1.0)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=2)
        self.proc.stdout.close()
        self.proc.stderr.close()
        for p in self.patches:
            p.stop()

    def test_address_file_holds_port_and_secret_token(self):
        self.assertIsInstance(self.addr["port"], int)
        self.assertGreaterEqual(len(self.addr["token"]), 64)
        if os.name == "posix":
            self.assertEqual(os.stat(self.addr_path).st_mode & 0o777, 0o600)

    def test_listens_on_loopback_only(self):
        with open(os.path.join(self.tmp, "daemon.log"), encoding="utf-8") as f:
            self.assertIn(f"listening on 127.0.0.1:{self.addr['port']}", f.read())
        self.assertEqual(_raw_request(self.addr["port"], {"op": "ping", "token": self.addr["token"]}),
                         {"ok": True})

    def test_requests_without_the_token_are_rejected(self):
        self.assertEqual(_raw_request(self.addr["port"], {"op": "ping"}), {"error": "unauthorized"})
        self.assertEqual(_raw_request(self.addr["port"], {"op": "shutdown", "token": "0" * 64}),
                         {"error": "unauthorized"})
        self.assertIsNone(self.proc.poll(), "a forged shutdown must not stop the daemon")

    def test_client_helpers_round_trip(self):
        resp = cv._send_to_daemon({"op": "status"})
        self.assertTrue(resp["ok"])
        # Not self.proc.pid: a Windows venv python.exe is a launcher process.
        with open(os.path.join(self.tmp, "daemon.pid"), encoding="utf-8") as f:
            self.assertEqual(resp["pid"], int(f.read()))

    def test_shutdown_removes_the_address_file(self):
        self.assertEqual(cv._send_to_daemon({"op": "shutdown"}), {"ok": True})
        self.proc.wait(timeout=5)
        self.assertFalse(os.path.exists(self.addr_path))
        self.assertFalse(cv._daemon_alive())


class LockFileTests(unittest.TestCase):
    def test_lock_is_exclusive_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.lock")
            holder = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys, time, claude_voice as cv\n"
                 f"f = open({path!r}, 'a')\n"
                 "cv._lock_file(f)\n"
                 "print('locked', flush=True)\n"
                 "sys.stdin.readline()\n"],
                env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), "locked")
                with open(path, "a") as f:
                    with self.assertRaises(OSError):
                        cv._lock_file(f, blocking=False)
                holder.stdin.write("\n")
                holder.stdin.flush()
                holder.wait(timeout=5)
                with open(path, "a") as f:
                    cv._lock_file(f, blocking=False)
                    cv._unlock_file(f)
            finally:
                holder.kill()
                holder.wait(timeout=5)
                holder.stdin.close()
                holder.stdout.close()


class WindowsBranchTests(unittest.TestCase):
    """Windows-only logic, exercised on any OS by patching the flag."""

    def test_terminal_key_is_shared_console(self):
        with mock.patch.object(cv, "WINDOWS", True):
            self.assertEqual(cv._resolve_tty(), cv.WINDOWS_TTY)

    def test_karaoke_writes_to_this_console_not_a_file_named_after_the_tty(self):
        with (
            mock.patch.object(cv, "WINDOWS", True),
            mock.patch.object(cv, "open", create=True, return_value=io.StringIO()) as fake_open,
        ):
            cv.get_tty(cv.WINDOWS_TTY)
        fake_open.assert_called_once_with("CONOUT$", "w")

    def test_hook_command_survives_git_bash(self):
        exe = r"C:\Users\Jan Kowalski(1)\venv\Scripts\claude-voice.exe"
        with (
            mock.patch.object(cv, "WINDOWS", True),
            mock.patch.object(cv.shutil, "which", return_value=exe),
        ):
            self.assertEqual(cv._invoker(),
                             '"C:/Users/Jan Kowalski(1)/venv/Scripts/claude-voice.exe"')

    def test_setup_writes_valid_frontmatter_for_a_quoted_invoker(self):
        invoker = '"C:/Users/Jan Kowalski(1)/claude-voice.exe"'
        with tempfile.TemporaryDirectory() as tmp:
            command = os.path.join(tmp, "commands", "voice.md")
            settings = os.path.join(tmp, "settings.json")
            with (
                mock.patch.object(cv, "_config", None),
                mock.patch.object(cv, "CONFIG_PATH", os.path.join(tmp, "config.json")),
                mock.patch.object(cv, "SETTINGS_PATH", settings),
                mock.patch.object(cv, "COMMAND_PATH", command),
                mock.patch.object(cv, "_invoker", return_value=invoker),
                mock.patch("sys.stdout", io.StringIO()),
            ):
                cv.cmd_setup()
            with open(command, encoding="utf-8") as f:
                text = f.read()
            with open(settings, encoding="utf-8") as f:
                hooks = json.load(f)["hooks"]
        allowed = next(line for line in text.splitlines() if line.startswith("allowed-tools:"))
        self.assertEqual(json.loads(allowed.split(":", 1)[1]), f"Bash({invoker} slash:*)")
        self.assertIn(f"!`{invoker} slash $ARGUMENTS`", text)
        self.assertEqual(hooks["Stop"][0]["hooks"][0]["command"], invoker)
        self.assertEqual(hooks["UserPromptSubmit"][0]["hooks"][0]["command"], f"{invoker} claim")
        context = hooks["UserPromptSubmit"][1]["hooks"][0]
        self.assertEqual(context["command"], f"{invoker} context")
        # Claude Code ignores the output of async hooks, so this one must block.
        self.assertNotIn("async", context)

    def test_system_voice_text_never_reaches_the_command_line(self):
        evil = "Done.'; Remove-Item -Recurse C:\\ ; $(calc) `whoami` \" #"
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            with wave.open(kwargs["env"]["CV_OUT"], "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(b"\x00\x00" * 100)
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        with (
            mock.patch.object(cv, "WINDOWS", True),
            mock.patch.object(cv, "_powershell", return_value="powershell"),
            mock.patch.object(cv.subprocess, "run", side_effect=fake_run),
        ):
            audio, rate, _ = cv.synth_system(evil, "Microsoft Paulina", 1.5, cv.default_config())

        cmd, kwargs = calls[0]
        self.assertEqual(cmd[:4], ["powershell", "-NoProfile", "-NonInteractive", "-Command"])
        self.assertEqual(cmd[4], cv._SAPI_SCRIPT)
        self.assertNotIn("Remove-Item", " ".join(cmd))
        self.assertIn("Remove-Item", kwargs["input"].decode("utf-8"))
        self.assertEqual(kwargs["env"]["CV_VOICE"], "Microsoft Paulina")
        self.assertEqual(kwargs["env"]["CV_RATE"], "5")
        self.assertEqual((len(audio), rate), (100, 22050))

    def test_daemon_never_draws_karaoke_on_windows(self):
        class Conn:
            def __init__(self, request):
                self.data, self.sent = json.dumps(request).encode() + b"\n", b""

            def settimeout(self, _):
                pass

            def recv(self, _):
                data, self.data = self.data, b""
                return data

            def sendall(self, data):
                self.sent += data

            def close(self):
                pass

        with (
            mock.patch.object(cv, "WINDOWS", True),
            mock.patch.object(cv, "DAEMON_TCP", False),
            mock.patch.object(cv, "reload_config", return_value=cv.default_config()),
            mock.patch.object(cv, "speak_and_highlight") as speak,
            mock.patch.object(cv, "_daemon_log"),
            mock.patch.object(cv, "_player", None),
        ):
            cv._handle_client(Conn({"op": "speak", "text": "hi", "render": True,
                                    "override": True, "tty_path": cv.WINDOWS_TTY}))
        self.assertFalse(speak.call_args.kwargs["render"])


class PathAndEncodingTests(unittest.TestCase):
    def test_project_paths_match_on_directory_boundaries(self):
        root = os.path.join(os.sep, "work", "app")
        self.assertTrue(cv._path_within(root, root))
        self.assertTrue(cv._path_within(os.path.join(root, "src"), root))
        self.assertFalse(cv._path_within(root + "2", root))

    @unittest.skipUnless(sys.platform == "win32", "Windows path semantics")
    def test_windows_project_paths_ignore_case_and_slash_style(self):
        self.assertTrue(cv._path_within(r"c:\projekty\app\lib", r"C:\Projekty\App"))
        self.assertTrue(cv._path_within("C:/Projekty/App/lib", r"C:\Projekty\App"))
        self.assertFalse(cv._path_within(r"C:\Projekty\App2", r"C:\Projekty\App"))

    @unittest.skipUnless(sys.platform == "win32", "Windows ACLs")
    def test_private_dirs_are_closed_to_other_accounts(self):
        with tempfile.TemporaryDirectory() as tmp:
            private = os.path.join(tmp, "runtime")
            os.makedirs(private)
            with mock.patch.object(cv, "_restricted_dirs", set()):
                cv._restrict_to_user(private)
            with open(os.path.join(private, "secret"), "w") as f:
                f.write("token")
            for path in (private, os.path.join(private, "secret")):
                acl = subprocess.run(["icacls", path], capture_output=True, text=True).stdout
                # "<path> DOMAIN\user:(OI)(CI)(F)" then one indented line per entry
                grants = [line.split(":(")[0].strip()
                          for line in acl.replace(path, "", 1).splitlines() if ":(" in line]
                self.assertEqual(grants, [f"{os.environ['USERDOMAIN']}\\{os.environ['USERNAME']}"],
                                 acl)

    def test_transcript_is_read_as_utf8(self):
        text = "Zażółć gęślą jaźń — gotowe ✓"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "t.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "assistant", "message": {
                    "content": [{"type": "text", "text": text}]}}, ensure_ascii=False) + "\n")
            got = cv.extract_hook_text(json.dumps({"transcript_path": path}))
        self.assertEqual(got, text)


if __name__ == "__main__":
    unittest.main()

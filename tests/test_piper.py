"""Piper provider: voice names, model download and synthesis (Piper faked)."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import claude_voice as cv


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PiperTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="cv-piper-test-")
        patches = [mock.patch.object(cv, "PIPER_DIR", self._tmp.name),
                   mock.patch.object(cv, "_piper_voices", {})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)


class VoiceNameTests(PiperTestCase):
    def test_valid_names_map_to_files_in_the_piper_dir(self):
        for name in ("pl_PL-gosia-medium", "pl_PL-mc_speech-medium", "en_US-lessac-x_low"):
            _, model, config = cv._piper_paths(name)
            self.assertEqual(model, os.path.join(self._tmp.name, name + ".onnx"))
            self.assertEqual(config, model + ".json")

    def test_names_that_could_escape_the_dir_or_url_are_rejected(self):
        for bad in ("../../evil-x-medium", "pl_PL-gosia-medium/../../x", "pl_PL-gosia",
                    "pl_PL-gosia-medium?x=1", "PL_pl-gosia-medium", "pl_PL-Gosia-medium",
                    "pl_PL-gosia-ultra", ""):
            with self.subTest(bad=bad), self.assertRaises(cv.ProviderError):
                cv._piper_paths(bad)


class DownloadTests(PiperTestCase):
    def test_downloads_model_and_config_from_the_official_repo_once(self):
        urls = []

        def urlopen(url, timeout):
            urls.append(url)
            return FakeResponse(b"model" if url.split("?")[0].endswith(".onnx") else b"{}")

        with mock.patch.object(cv.urllib.request, "urlopen", side_effect=urlopen):
            cv._piper_download("pl_PL-gosia-medium")
            cv._piper_download("pl_PL-gosia-medium")

        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/"
        self.assertEqual(urls, [base + "pl_PL-gosia-medium.onnx.json?download=true",
                                base + "pl_PL-gosia-medium.onnx?download=true"])
        self.assertEqual(sorted(os.listdir(self._tmp.name)),
                         ["pl_PL-gosia-medium.onnx", "pl_PL-gosia-medium.onnx.json"])

    def test_interrupted_download_leaves_no_file_that_looks_finished(self):
        class Broken(FakeResponse):
            def read(self, *a):
                raise OSError("connection reset")

        with (
            mock.patch.object(cv.urllib.request, "urlopen", return_value=Broken(b"")),
            self.assertRaises(cv.ProviderError),
        ):
            cv._piper_download("pl_PL-gosia-medium")
        self.assertEqual(os.listdir(self._tmp.name), [])

    def test_unknown_voice_reports_a_clear_error(self):
        err = cv.urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        with (
            mock.patch.object(cv.urllib.request, "urlopen", side_effect=err),
            self.assertRaisesRegex(cv.ProviderError, "not found"),
        ):
            cv._piper_download("pl_PL-nobody-medium")


class SynthesisTests(PiperTestCase):
    def fake_piper(self):
        calls = []

        class Voice:
            def synthesize(self, text, syn_config=None):
                calls.append((text, syn_config))
                for n in (100, 50):
                    yield SimpleNamespace(audio_float_array=np.full(n, 0.5, dtype=np.float32),
                                          sample_rate=22050)

        piper = types.ModuleType("piper")
        piper.PiperVoice = SimpleNamespace(load=lambda model, config_path=None: Voice())
        config = types.ModuleType("piper.config")
        config.SynthesisConfig = lambda **kw: SimpleNamespace(**kw)
        modules = mock.patch.dict(sys.modules, {"piper": piper, "piper.config": config})
        modules.start()
        self.addCleanup(modules.stop)
        no_download = mock.patch.object(cv, "_piper_download")
        no_download.start()
        self.addCleanup(no_download.stop)
        return calls

    def test_audio_chunks_are_joined_at_the_voice_rate(self):
        self.fake_piper()
        audio, rate, timings = cv.synth_piper("Cześć.", "pl_PL-gosia-medium", 1.0,
                                              cv.default_config())
        self.assertEqual((len(audio), rate, timings), (150, 22050, None))

    def test_speed_maps_to_piper_length_scale(self):
        calls = self.fake_piper()
        cv.synth_piper("Cześć.", "pl_PL-gosia-medium", 1.25, cv.default_config())
        self.assertAlmostEqual(calls[0][1].length_scale, 0.8)

    def test_english_dev_pronunciations_only_apply_to_english_voices(self):
        calls = self.fake_piper()
        cfg = cv.default_config()
        cv.synth_piper("Plik JSON.", "pl_PL-gosia-medium", 1.0, cfg)
        cv.synth_piper("The JSON file.", "en_US-lessac-medium", 1.0, cfg)
        self.assertEqual(calls[0][0], "Plik JSON.")
        self.assertIn("jason", calls[1][0])

    def test_loaded_voice_is_reused(self):
        self.fake_piper()
        first = cv.get_piper_voice("pl_PL-gosia-medium")
        self.assertIs(cv.get_piper_voice("pl_PL-gosia-medium"), first)


class CommandTests(PiperTestCase):
    def test_slash_rejects_a_malformed_piper_voice(self):
        cfg = dict(cv.default_config(), provider="piper")
        with (
            mock.patch.object(cv, "load_config", return_value=cfg),
            mock.patch.object(cv, "save_config") as save,
            mock.patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            cv.cmd_slash(["voice", "../../evil"])
        self.assertIn("error:", out.getvalue())
        save.assert_not_called()

    def test_voices_lists_downloaded_models(self):
        open(os.path.join(self._tmp.name, "pl_PL-darkman-medium.onnx"), "w").close()
        open(os.path.join(self._tmp.name, "uk_UA-lada-x_low.onnx"), "w").close()
        with (
            mock.patch.object(cv, "load_config", return_value=cv.default_config()),
            mock.patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            cv.cmd_voices(["piper"])
        self.assertIn("pl_PL-gosia-medium", out.getvalue())
        self.assertIn("uk_UA-lada-x_low", out.getvalue())


if __name__ == "__main__":
    unittest.main()

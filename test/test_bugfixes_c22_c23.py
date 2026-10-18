"""Regression tests: a missing/corrupt output-dir config must not crash generation.

Background: a stray test once wrote a pytest tmp path into the user's real
user_data/config.json (txt2img_dir). ConfigManager.get() returns None when
a _dir path does not exist, which made get_next_media (os.path.isdir(None)
-> "stat: ... not NoneType") and _set_output_path (os.path.join(None, ...)
-> "expected str, bytes or os.PathLike ... not NoneType") crash.

Fixes locked in here:
- ConfigManager.get_or_create_dir() never returns None for dir keys
  (falls back to DEFAULT_SETTINGS, creates the directory).
- test/conftest.py isolates SD_WEBUI_CONFIG_PATH at module import time,
  so the shared_instance config singleton can never bind to the user's
  real config during tests.
"""
import os

from modules.config import ConfigManager
from modules.shared_instance import config


class TestGetOrCreateDir:
    def test_creates_missing_dir(self, tmp_path):
        cm = ConfigManager(tmp_path / "config.json")
        target = tmp_path / "never" / "created"
        cm.update_settings({"txt2img_dir": str(target)})
        d = cm.get_or_create_dir("txt2img_dir")
        assert d == str(target)
        assert os.path.isdir(d)

    def test_returns_existing_dir(self, tmp_path):
        cm = ConfigManager(tmp_path / "config.json")
        target = tmp_path / "exists"
        target.mkdir()
        cm.update_settings({"txt2img_dir": str(target)})
        assert cm.get_or_create_dir("txt2img_dir") == str(target)

    def test_invalid_value_falls_back_to_default(self, tmp_path):
        cm = ConfigManager(tmp_path / "config.json")
        cm.update_settings({"txt2img_dir": ""})
        d = cm.get_or_create_dir("txt2img_dir")
        assert d and os.path.isdir(d)

    def test_missing_key_uses_default(self, tmp_path):
        cm = ConfigManager(tmp_path / "config.json")
        d = cm.get_or_create_dir("txt2img_dir")
        assert d and os.path.isdir(d)


class TestGenerationSurvivesMissingOutputDir:
    """The exact user crashes, replayed through the real code paths."""

    @staticmethod
    def _isolated_dir(name):
        return os.path.join(os.path.dirname(os.environ["SD_WEBUI_CONFIG_PATH"]), name)

    def test_set_output_path_named(self):
        from modules.core.server.sdcpp_server import Txt2ImgApiRunner
        original = config.data.get("txt2img_dir")
        try:
            target = self._isolated_dir("broken_named_dir")
            config.update_settings({"txt2img_dir": target})
            run = Txt2ImgApiRunner(
                {"in_ip": "127.0.0.1", "in_port": "7860", "in_output": "regress"}
            )
            run.prepare()
            assert run.output_path
            assert os.path.isdir(target)
        finally:
            config.update_settings({"txt2img_dir": original})

    def test_set_output_path_anonymous(self):
        from modules.core.server.sdcpp_server import Txt2ImgApiRunner
        original = config.data.get("txt2img_dir")
        try:
            target = self._isolated_dir("broken_anon_dir")
            config.update_settings({"txt2img_dir": target})
            run = Txt2ImgApiRunner({"in_ip": "127.0.0.1", "in_port": "7860"})
            run.prepare()
            assert run.output_path.endswith(".png")
            assert os.path.isdir(target)
        finally:
            config.update_settings({"txt2img_dir": original})

    def test_get_next_media_survives_missing_dir(self):
        from modules.gallery import get_next_media
        original = config.data.get("txt2img_dir")
        try:
            target = self._isolated_dir("broken_media_dir")
            config.update_settings({"txt2img_dir": target})
            result = get_next_media(0)
            assert result == "1.png"
            assert os.path.isdir(target)
        finally:
            config.update_settings({"txt2img_dir": original})

import os
import shutil
import tempfile
from pathlib import Path

# IMPORTANT: all isolation happens at MODULE level, before any app module
# is imported. modules/shared_instance.py builds `config = ConfigManager()`
# at import time. Depending on how a test module imports app code, that
# import can happen during collection (module-level import) or at first
# fixture use (function-level import). In BOTH cases the env vars and the
# config file must already exist, or the singleton loads from (or binds to)
# the user's real user_data/config.json and tests can corrupt it.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="sdcpp-webui-test-"))

for _sub in ("txt2img", "img2img"):
    (_TEST_ROOT / _sub).mkdir()

os.environ['SD_WEBUI_CONFIG_PATH'] = str(_TEST_ROOT / "config.json")
os.environ['SD_WEBUI_PROMPTS_PATH'] = str(_TEST_ROOT / "prompts.json")

import pytest  # noqa: E402

from modules.config import ConfigManager  # noqa: E402

# Seed the config file with the isolated output dirs so that however and
# whenever the shared singleton is constructed, it loads these values.
ConfigManager(os.environ['SD_WEBUI_CONFIG_PATH']).update_settings({
    "txt2img_dir": str(_TEST_ROOT / "txt2img"),
    "img2img_dir": str(_TEST_ROOT / "img2img"),
})


@pytest.fixture(autouse=True, scope="session")
def app_root():
    """Temporary application root (created at module level, see above)."""
    yield _TEST_ROOT
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def sd_options_mock(mocker):
    mocker.patch("modules.utils.sd_interface.SDOptionsCache")

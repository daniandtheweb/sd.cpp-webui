import pytest
import os

from modules.config import ConfigManager

@pytest.fixture(autouse=True, scope="session")
def app_root(tmp_path_factory):
    """Set up a temporary application root with config files and output directories."""
    tmp_path = tmp_path_factory.mktemp("sdcpp-webui")
    config_path = tmp_path / "config.json"
    prompts_path = tmp_path / "prompts.json"

    # output directories
    txt2img_dir = tmp_path / "txt2img"
    txt2img_dir.mkdir()
    img2img_dir = tmp_path / "img2img"
    img2img_dir.mkdir()

    # Initialize default config files
    config = ConfigManager(config_path, prompts_path)
    config.update_settings({
        "txt2img_dir": str(txt2img_dir),
        "img2img_dir": str(img2img_dir),
    })

    # Export environment variables for the application to use
    os.environ['SD_WEBUI_CONFIG_PATH'] = str(config_path)
    os.environ['SD_WEBUI_PROMPTS_PATH'] = str(prompts_path)

    yield tmp_path

    del os.environ['SD_WEBUI_CONFIG_PATH']
    del os.environ['SD_WEBUI_PROMPTS_PATH']


@pytest.fixture(autouse=True)
def sd_options_mock(mocker):
    mocker.patch("modules.utils.sd_interface.SDOptionsCache")

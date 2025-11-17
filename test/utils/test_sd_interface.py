import os
import pytest
from modules.utils.sd_interface import SDOptionsCache

SD_VERSION = "master-362-742a733"


@pytest.fixture(autouse=True)
def app_cwd(tmp_path, monkeypatch):
    """Change to clean working directory to avoid polluting project root with cache files."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture(autouse=True)
def sd_binary(request, mocker) -> None:
    """Mock sd binary help output and path."""
    help_output = (request.path.parent / f"{SD_VERSION}.txt").read_text()

    shutil_which_mock = mocker.patch("modules.utils.sd_interface.shutil.which")
    shutil_which_mock.return_value = "/usr/local/bin/sd"

    subprocess_run_mock = mocker.patch("modules.utils.sd_interface.subprocess.run")
    subprocess_run_mock.return_value.configure_mock(
        stdout=help_output,
        returncode=0,
    )


def test_options_cache(app_cwd):
    sd_options = SDOptionsCache()
    assert sd_options.SD_PATH == "/usr/local/bin/sd"
    assert (app_cwd / sd_options._CACHE_FILE).exists()


@pytest.mark.parametrize(
    "option,expected",
    [
        (
            "samplers",
            [
                "euler",
                "euler_a",
                "heun",
                "dpm2",
                "dpm++2s_a",
                "dpm++2m",
                "dpm++2mv2",
                "ipndm",
                "ipndm_v",
                "lcm",
                "ddim_trailing",
                "tcd",
            ],
        ),
        (
            "schedulers",
            [
                "discrete",
                "karras",
                "exponential",
                "ays",
                "gits",
                "smoothstep",
                "sgm_uniform",
                "simple",
            ],
        ),
        ("previews", ["none", "proj", "tae", "vae"]),
        ("prediction", ["eps", "v", "edm_v", "sd3_flow", "flux_flow"]),
        ("rng", ["std_default", "cuda", "cpu"]),
        ("sampler_rng", []),
    ],
)
def test_options_parse(option, expected) -> None:
    sd_options = SDOptionsCache()
    assert sd_options.get_opt(option) == expected

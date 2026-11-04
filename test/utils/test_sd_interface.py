import json
import os
import pytest
from modules.utils.sd_interface import SDChoicesCache

SD_VERSION = "master-367-b88cc32"


@pytest.fixture(autouse=True)
def app_cwd(tmp_path, monkeypatch):
    """Change to clean working directory to avoid polluting project root with cache files."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path


@pytest.fixture(autouse=True)
def sd_binary(request, mocker) -> None:
    """Mock sd binary help output and path."""
    help_output = (request.path.parent / f"{SD_VERSION}.txt").read_text()

    def which_side_effect(name):
        return "/usr/local/bin/sd" if name == "sd" else None

    shutil_which_mock = mocker.patch(
        "modules.utils.sd_interface.shutil.which",
        side_effect=which_side_effect,
    )

    subprocess_run_mock = mocker.patch("modules.utils.sd_interface.subprocess.run")
    subprocess_run_mock.return_value.configure_mock(
        stdout=help_output,
        returncode=0,
    )


def test_choices_cache(app_cwd):
    sd_choices = SDChoicesCache()
    assert sd_choices.SD_PATH == "sd"
    assert (app_cwd / sd_choices._CACHE_FILE).exists()


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
        ("sampler_rng", ["std_default", "cuda", "cpu"]),
    ],
)
def test_choices_parse(option, expected) -> None:
    sd_choices = SDChoicesCache()
    assert sd_choices.get_choices(option) == expected


def test_an_empty_help_output_is_not_cached(app_cwd, mocker):
    """A failed --help must not leave a cache that hides the real choices."""
    mocker.patch(
        "modules.utils.sd_interface.subprocess.run"
    ).return_value.configure_mock(stdout="", returncode=0)
    sd_choices = SDChoicesCache()
    assert not (app_cwd / sd_choices._CACHE_FILE).exists()
    # the built-in fallbacks are used meanwhile
    assert "euler_a" in sd_choices.get_choices("samplers")


def test_a_cache_without_choices_is_rebuilt(app_cwd):
    """An empty cache (right hashes, no choices) is not trusted."""
    (app_cwd / "sd").write_bytes(b"binary")       # so the cache is checked
    cache = app_cwd / SDChoicesCache()._CACHE_FILE
    data = json.loads(cache.read_text())
    assert data["choices"]["--sampling-method"]
    data["choices"] = {flag: [] for flag in data["choices"]}
    cache.write_text(json.dumps(data))

    rebuilt = SDChoicesCache()
    assert rebuilt._help_cache["--sampling-method"]
    assert json.loads(cache.read_text())["choices"]["--sampling-method"]


def test_a_good_cache_is_kept(app_cwd, mocker):
    """A valid cache is read, the --help command isn't run again."""
    (app_cwd / "sd").write_bytes(b"binary")
    SDChoicesCache()
    run = mocker.patch("modules.utils.sd_interface.subprocess.run")
    cached = SDChoicesCache()
    assert cached._help_cache["--sampling-method"]
    help_calls = [c for c in run.call_args_list if "--help" in c.args[0]]
    assert not help_calls

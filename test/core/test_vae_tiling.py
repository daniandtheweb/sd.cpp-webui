"""VAE tiling options in the image and video sd-cli commands."""
import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


_TILING = {
    'in_vae_tiling': True,
    'in_vae_tile_overlap': 0.5,
    'in_vae_tile_size': 256.0,
    'in_vae_relative_bool': True,
    'in_vae_relative_tile_size': 2.0,
}


def _command(tmp_path, monkeypatch, video, **params):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner, Any2VideoRunner
    base = {
        'in_pprompt': 'a lovely cat',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 832,
        'in_height': 480,
        'in_cfg': 6.0,
        'in_seed': 42,
        'in_frames': 33,
        'in_fps': 16,
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    if video:
        runner = Any2VideoRunner(mode='vid_gen', params=base)
    else:
        runner = Txt2ImgRunner('txt2img', base)
    runner.build_command()
    return runner.command


@pytest.mark.parametrize('video', [False, True])
def test_tiling_options(tmp_path, monkeypatch, video):
    command = _command(tmp_path, monkeypatch, video, **_TILING)
    assert '--vae-tiling' in command
    assert _flag(command, '--vae-tile-overlap') == '0.5'
    assert _flag(command, '--vae-tile-size') == '256x256'
    assert _flag(command, '--vae-relative-tile-size') == '2x2'


@pytest.mark.parametrize('video', [False, True])
def test_tiling_off(tmp_path, monkeypatch, video):
    command = _command(
        tmp_path, monkeypatch, video, **{**_TILING, 'in_vae_tiling': False}
    )
    for flag in ('--vae-tiling', '--vae-tile-overlap', '--vae-tile-size',
                 '--vae-relative-tile-size'):
        assert flag not in command

"""Extra sample and tiling args: sd-cli, sd-server launch and payload."""
import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


_EXTRA_ARGS = {
    'in_extra_sample_args': ' apg_eta=0.5,apg_momentum=-0.5 ',
    'in_extra_tiling_args': 'temporal_tile_frames=4,temporal_decode_frames=2',
}


def _base_params():
    return {
        'in_pprompt': 'a lovely cat',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 6.0,
        'in_seed': 42,
    }


def test_cli_image(tmp_path, monkeypatch):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Txt2ImgRunner('txt2img', {**_base_params(), **_EXTRA_ARGS})
    runner.build_command()
    assert _flag(runner.command, '--extra-sample-args') == (
        'apg_eta=0.5,apg_momentum=-0.5'
    )
    assert _flag(runner.command, '--extra-tiling-args') == (
        'temporal_tile_frames=4,temporal_decode_frames=2'
    )


def test_cli_video(tmp_path, monkeypatch):
    from modules.core.cli.sdcpp_cli import Any2VideoRunner
    monkeypatch.chdir(tmp_path)
    runner = Any2VideoRunner(mode='vid_gen', params={
        **_base_params(), **_EXTRA_ARGS, 'in_frames': 33, 'in_fps': 16,
    })
    runner.build_command()
    assert '--extra-sample-args' in runner.command
    assert '--extra-tiling-args' in runner.command


@pytest.mark.parametrize('value', ['', '  ', None])
def test_cli_empty_is_omitted(tmp_path, monkeypatch, value):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    monkeypatch.chdir(tmp_path)
    runner = Txt2ImgRunner('txt2img', {
        **_base_params(),
        'in_extra_sample_args': value, 'in_extra_tiling_args': value,
    })
    runner.build_command()
    assert '--extra-sample-args' not in runner.command
    assert '--extra-tiling-args' not in runner.command


def test_server_sends_them_per_request():
    from modules.core.server.manager import ServerRunner
    from modules.core.server import sdcpp_server
    runner = ServerRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234, **_EXTRA_ARGS,
    })
    runner.build_command()
    assert '--extra-sample-args' not in runner.command
    assert '--extra-tiling-args' not in runner.command

    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_vae_tiling': False, **_EXTRA_ARGS,
    })._build_payload()
    assert payload['sample_params']['extra_sample_args'] == (
        'apg_eta=0.5,apg_momentum=-0.5'
    )
    # Sent even without spatial tiling, like temporal tiling
    assert payload['vae_tiling_params']['extra_tiling_args'] == (
        'temporal_tile_frames=4,temporal_decode_frames=2'
    )


def test_server_payload_empty():
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234, 'in_vae_tiling': True,
        'in_extra_sample_args': '', 'in_extra_tiling_args': ' ',
    })._build_payload()
    assert 'extra_sample_args' not in payload['sample_params']
    assert 'extra_tiling_args' not in payload['vae_tiling_params']

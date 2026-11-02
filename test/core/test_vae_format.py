"""VAE format override: sd-cli and sd-server commands."""
import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _cli_command(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    base = {
        'in_pprompt': 'a cat',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 7.0,
        'in_seed': 42,
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Txt2ImgRunner('txt2img', base)
    runner.build_command()
    return runner.command


def test_cli_override(tmp_path, monkeypatch):
    command = _cli_command(tmp_path, monkeypatch, in_vae_format='flux2')
    assert _flag(command, '--vae-format') == 'flux2'


@pytest.mark.parametrize('value', ['Default', None])
def test_cli_default_is_omitted(tmp_path, monkeypatch, value):
    command = _cli_command(tmp_path, monkeypatch, in_vae_format=value)
    assert '--vae-format' not in command


def test_server_launch_override():
    from modules.core.server.manager import ServerRunner
    runner = ServerRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234, 'in_vae_format': 'wan',
    })
    runner.build_command()
    assert _flag(runner.command, '--vae-format') == 'wan'

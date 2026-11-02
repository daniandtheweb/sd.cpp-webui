"""PuLID: sd-cli command, sd-server launch, embedding listing."""
import os

import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _pulid_params(**params):
    base = {
        'in_pulid_bool': True,
        'in_pulid_weights': 'pulid_flux_v0.9.1.safetensors',
        'in_pulid_embedding': 'person.pulidembd',
        'in_pulid_id_weight': 0.9,
    }
    base.update(params)
    return base


def _cli_command(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    base = {
        'in_pprompt': 'candid photograph of a woman on a beach',
        'in_nprompt': '',
        'in_steps': 4,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 1.0,
        'in_seed': 42,
    }
    base.update(_pulid_params(**params))
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Txt2ImgRunner('txt2img', base)
    runner.build_command()
    return runner.command


def test_cli_enabled(tmp_path, monkeypatch):
    command = _cli_command(tmp_path, monkeypatch)
    assert os.path.basename(_flag(command, '--pulid-weights')) == (
        'pulid_flux_v0.9.1.safetensors'
    )
    assert os.path.basename(_flag(command, '--pulid-id-embedding')) == (
        'person.pulidembd'
    )
    assert _flag(command, '--pulid-id-weight') == '0.9'


@pytest.mark.parametrize('params', [
    {'in_pulid_bool': False},
    {'in_pulid_weights': None},
    {'in_pulid_embedding': None},
])
def test_cli_inactive(tmp_path, monkeypatch, params):
    command = _cli_command(tmp_path, monkeypatch, **params)
    assert not any(str(arg).startswith('--pulid') for arg in command)


def test_server_launch():
    from modules.core.server.manager import ServerRunner
    runner = ServerRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234, **_pulid_params(),
    })
    runner.build_command()
    # sd-server has no per-request PuLID fields, all three are launch options
    assert os.path.basename(_flag(runner.command, '--pulid-weights')) == (
        'pulid_flux_v0.9.1.safetensors'
    )
    assert os.path.basename(
        _flag(runner.command, '--pulid-id-embedding')
    ) == 'person.pulidembd'
    assert _flag(runner.command, '--pulid-id-weight') == '0.9'


def test_embeddings_listed_apart_from_weights(tmp_path):
    from modules.loader import get_models
    from modules.ui.constants import PULID_EMBEDDING_EXT
    for name in ('pulid_flux_v0.9.1.safetensors', 'person.pulidembd'):
        (tmp_path / name).write_bytes(b'')
    assert get_models(str(tmp_path)) == ['pulid_flux_v0.9.1.safetensors']
    assert get_models(str(tmp_path), PULID_EMBEDDING_EXT) == [
        'person.pulidembd'
    ]

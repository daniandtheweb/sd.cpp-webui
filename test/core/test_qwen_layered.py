"""Qwen Image Layered: sd-cli command and outputs, sd-server payload."""
import os

import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _imgedit_runner(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import ImgEditRunner
    base = {
        'in_pprompt': 'split the image into layers',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 640,
        'in_height': 640,
        'in_cfg': 4.0,
        'in_seed': 42,
        'in_ref_img': [str(tmp_path / 'reference.png')],
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = ImgEditRunner('img_gen', base)
    runner.build_command()
    runner._prepare_for_run()
    return runner


def _names(runner):
    return [os.path.basename(path) for path in runner.outputs]


def test_cli_layers_and_outputs(tmp_path, monkeypatch):
    runner = _imgedit_runner(
        tmp_path, monkeypatch,
        in_qwen_layered_bool=True, in_qwen_image_layers=3.0,
    )
    assert _flag(runner.command, '--qwen-image-layers') == '3'
    # layers + 1 images, saved like a batch: <base>_0 ... <base>_3
    base = os.path.splitext(os.path.basename(runner.output_path))[0]
    assert _names(runner) == [f"{base}_{i}.png" for i in range(4)]


def test_cli_outputs_with_batch(tmp_path, monkeypatch):
    runner = _imgedit_runner(
        tmp_path, monkeypatch,
        in_qwen_layered_bool=True, in_qwen_image_layers=2,
        in_batch_count=2,
    )
    assert len(runner.outputs) == 6


def test_cli_zero_layers_single_output(tmp_path, monkeypatch):
    runner = _imgedit_runner(
        tmp_path, monkeypatch,
        in_qwen_layered_bool=True, in_qwen_image_layers=0,
    )
    assert _flag(runner.command, '--qwen-image-layers') == '0'
    assert runner.outputs == [runner.output_path]


@pytest.mark.parametrize('params', [
    {'in_qwen_layered_bool': False, 'in_qwen_image_layers': 3},
    {},
])
def test_cli_disabled(tmp_path, monkeypatch, params):
    runner = _imgedit_runner(tmp_path, monkeypatch, **params)
    assert '--qwen-image-layers' not in runner.command
    assert runner.outputs == [runner.output_path]


def test_server_payload():
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.ImgEditApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_qwen_layered_bool': True, 'in_qwen_image_layers': 5.0,
    })._build_payload()
    assert payload['qwen_image_layers'] == 5


def test_server_payload_disabled():
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.ImgEditApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_qwen_layered_bool': False, 'in_qwen_image_layers': 5,
    })._build_payload()
    assert 'qwen_image_layers' not in payload

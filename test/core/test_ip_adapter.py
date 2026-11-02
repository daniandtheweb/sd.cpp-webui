"""IP-Adapter: sd-cli command, sd-server launch and payload."""
import base64
import io
import os

from PIL import Image


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _image(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), (255, 128, 0)).save(path)
    return str(path)


def _cli_command(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    base = {
        'in_pprompt': 'a woman, best quality',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 7.0,
        'in_seed': 42,
        'in_ip_adapter': 'ip-adapter_sd15.safetensors',
        'in_ip_adapter_clip_vision': 'clip_vision_h.safetensors',
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Txt2ImgRunner('txt2img', base)
    runner.build_command()
    return runner.command


# ----------------------------------------------------------------------
# sd-cli
# ----------------------------------------------------------------------

def test_cli_enabled(tmp_path, monkeypatch):
    image = _image(tmp_path)
    command = _cli_command(
        tmp_path, monkeypatch,
        in_ip_adapter_bool=True, in_ip_adapter_img=image,
        in_ip_adapter_strength=0.8,
    )
    assert os.path.basename(_flag(command, '--ip-adapter')) == (
        'ip-adapter_sd15.safetensors'
    )
    assert os.path.basename(_flag(command, '--clip_vision')) == (
        'clip_vision_h.safetensors'
    )
    assert os.path.basename(_flag(command, '--ip-adapter-image')) == (
        'reference.png'
    )
    assert _flag(command, '--ip-adapter-strength') == '0.8'


def test_cli_disabled(tmp_path, monkeypatch):
    command = _cli_command(
        tmp_path, monkeypatch,
        in_ip_adapter_bool=False, in_ip_adapter_img=_image(tmp_path),
    )
    assert not any(str(arg).startswith('--ip-adapter') for arg in command)
    assert '--clip_vision' not in command


def test_cli_enabled_without_reference_image(tmp_path, monkeypatch):
    command = _cli_command(
        tmp_path, monkeypatch,
        in_ip_adapter_bool=True, in_ip_adapter_img=None,
    )
    assert not any(str(arg).startswith('--ip-adapter') for arg in command)
    assert '--clip_vision' not in command


# ----------------------------------------------------------------------
# sd-server
# ----------------------------------------------------------------------

def _server_command(**params):
    from modules.shared_instance import SD_CLI
    from modules.core.server.manager import ServerRunner
    exe_dir = os.path.dirname(os.path.abspath(SD_CLI))
    base = {'in_ip': '127.0.0.1', 'in_port': 1234}
    base.update({
        k: os.path.join(exe_dir, 'models', v) for k, v in params.items()
    })
    runner = ServerRunner(base)
    runner.build_command()
    return runner.command


def test_server_loads_adapter_and_encoder_at_launch():
    command = _server_command(
        f_ip_adapter='ip.safetensors',
        f_ip_adapter_clip_vision='clip_vision_h.safetensors',
    )
    assert _flag(command, '--ip-adapter') == 'models/ip.safetensors'
    assert _flag(command, '--clip_vision') == (
        'models/clip_vision_h.safetensors'
    )
    # The reference image and strength are sent per request
    assert '--ip-adapter-image' not in command


def test_server_skips_encoder_without_adapter():
    command = _server_command(
        f_ip_adapter_clip_vision='clip_vision_h.safetensors',
    )
    assert '--ip-adapter' not in command
    assert '--clip_vision' not in command


def test_server_payload(tmp_path):
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_ip_adapter_bool': True,
        'in_ip_adapter_img': _image(tmp_path),
        'in_ip_adapter_strength': 0.7,
    })._build_payload()
    assert payload['ip_adapter_strength'] == 0.7
    assert Image.open(io.BytesIO(
        base64.b64decode(payload['ip_adapter_image']))).size == (8, 8)


def test_server_payload_disabled(tmp_path):
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_ip_adapter_bool': False,
        'in_ip_adapter_img': _image(tmp_path),
    })._build_payload()
    assert 'ip_adapter_image' not in payload
    assert 'ip_adapter_strength' not in payload

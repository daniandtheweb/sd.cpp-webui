"""ADetailer: extra args, sd-cli command, sd-server launch and payload."""
import os


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _cli_params(**params):
    base = {
        'in_pprompt': 'portrait photo',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 7.0,
        'in_seed': 42,
        'in_ad_model': 'face_yolov8n.safetensors',
    }
    base.update(params)
    return base


def _cli_command(tmp_path, monkeypatch, params):
    from modules.core.cli.sdcpp_cli import Txt2ImgRunner
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Txt2ImgRunner('txt2img', params)
    runner.build_command()
    return runner.command


# ----------------------------------------------------------------------
# --extra-ad-args
# ----------------------------------------------------------------------

def test_extra_args_skip_sdcpp_defaults():
    from modules.core.common.sd_common import CommonRunner
    runner = CommonRunner({
        'in_ad_confidence': 0.3, 'in_ad_denoising_strength': 0.4,
        'in_ad_mask_blur': 4.0, 'in_ad_inpaint_padding': 32,
        'in_ad_dilate_erode': 4, 'in_ad_mask_mode': 'none',
        'in_ad_inpaint_width': 512, 'in_ad_inpaint_height': 512,
        'in_ad_steps': 0, 'in_ad_cfg': -1, 'in_ad_extra_args': ' ',
    })
    assert runner._build_extra_ad_args() is None


def test_extra_args_changed_values_then_free_form():
    from modules.core.common.sd_common import CommonRunner
    runner = CommonRunner({
        'in_ad_confidence': 0.45, 'in_ad_denoising_strength': 0.35,
        'in_ad_mask_blur': 8.0, 'in_ad_dilate_erode': -2,
        'in_ad_mask_mode': 'merge', 'in_ad_inpaint_width': 768.0,
        'in_ad_steps': 12, 'in_ad_cfg': 5.0,
        'in_ad_extra_args': ' mask_k_largest=1,sort_by=area, ',
    })
    assert runner._build_extra_ad_args() == (
        "confidence=0.45,denoising_strength=0.35,mask_blur=8,"
        "dilate_erode=-2,mask_mode=merge,inpaint_width=768,steps=12,"
        "cfg_scale=5,mask_k_largest=1,sort_by=area"
    )


# ----------------------------------------------------------------------
# sd-cli
# ----------------------------------------------------------------------

def test_cli_enabled(tmp_path, monkeypatch):
    command = _cli_command(tmp_path, monkeypatch, _cli_params(
        in_ad_bool=True, in_ad_prompt=' [PROMPT], detailed face ',
        in_ad_nprompt='', in_ad_confidence=0.5,
    ))
    assert os.path.basename(_flag(command, '--ad-model')) == (
        'face_yolov8n.safetensors'
    )
    assert _flag(command, '--ad-prompt') == '[PROMPT], detailed face'
    assert '--ad-negative-prompt' not in command
    assert _flag(command, '--extra-ad-args') == 'confidence=0.5'


def test_cli_disabled(tmp_path, monkeypatch):
    command = _cli_command(tmp_path, monkeypatch, _cli_params(
        in_ad_bool=False, in_ad_prompt='detailed face',
    ))
    assert not any(str(arg).startswith('--ad-') for arg in command)
    assert '--extra-ad-args' not in command


def test_cli_enabled_without_detector(tmp_path, monkeypatch):
    command = _cli_command(tmp_path, monkeypatch, _cli_params(
        in_ad_bool=True, in_ad_model=None, in_ad_prompt='detailed face',
    ))
    assert not any(str(arg).startswith('--ad-') for arg in command)


# ----------------------------------------------------------------------
# sd-server
# ----------------------------------------------------------------------

def test_server_loads_detector_at_launch():
    from modules.shared_instance import SD_CLI
    from modules.core.server.manager import ServerRunner
    exe_dir = os.path.dirname(os.path.abspath(SD_CLI))

    runner = ServerRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'f_ad_model': os.path.join(exe_dir, 'models', 'face.safetensors'),
    })
    runner.build_command()
    assert _flag(runner.command, '--ad-model') == 'models/face.safetensors'
    # Prompts and settings are sent per request
    assert '--ad-prompt' not in runner.command
    assert '--extra-ad-args' not in runner.command


def test_server_payload():
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
        'in_ad_prompt': 'detailed face', 'in_ad_nprompt': ' ',
        'in_ad_mask_mode': 'merge_invert',
    })._build_payload()
    assert payload['ad_prompt'] == 'detailed face'
    assert 'ad_negative_prompt' not in payload
    assert payload['extra_ad_args'] == 'mask_mode=merge_invert'


def test_server_payload_defaults_send_nothing():
    from modules.core.server import sdcpp_server
    payload = sdcpp_server.Txt2ImgApiRunner({
        'in_ip': '127.0.0.1', 'in_port': 1234,
    })._build_payload()
    assert not any(key.startswith('ad_') for key in payload)
    assert 'extra_ad_args' not in payload

"""AnimateDiff: the any2video command in the UNET and AnimateDiff modes."""
import os

import pytest


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _video_command(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import Any2VideoRunner
    base = {
        'in_pprompt': 'a red apple on a wooden table',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 8.0,
        'in_seed': 42,
        'in_frames': 16,
        'in_fps': 8,
        # Both tabs filled in
        'in_unet_model': 'wan2.1_t2v.gguf',
        'in_unet_vae': 'wan_vae.safetensors',
        'in_umt5_xxl': 'umt5_xxl.gguf',
        'in_high_noise_bool': True,
        'in_high_noise_model': 'wan2.2_high_noise.gguf',
        'in_motion_module': 'mm_sd15_v3.safetensors',
        'in_animatediff_ckpt': 'realisticVisionV60B1.safetensors',
        'in_animatediff_vae': 'vae-ft-mse-840000.safetensors',
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Any2VideoRunner(mode='vid_gen', params=base)
    runner.build_command()
    return runner.command


def test_animatediff_mode(tmp_path, monkeypatch):
    # The Gradio mode number arrives as a float
    command = _video_command(tmp_path, monkeypatch, in_video_mode=1.0)
    assert _flag(command, '-M') == 'vid_gen'
    assert os.path.basename(_flag(command, '--model')) == (
        'realisticVisionV60B1.safetensors'
    )
    assert os.path.basename(_flag(command, '--motion-module')) == (
        'mm_sd15_v3.safetensors'
    )
    # The checkpoint VAE replaces the video VAE
    assert os.path.basename(_flag(command, '--vae')) == (
        'vae-ft-mse-840000.safetensors'
    )
    assert _flag(command, '--video-frames') == '16'
    for flag in ('--diffusion-model', '--t5xxl',
                 '--high-noise-diffusion-model'):
        assert flag not in command


def test_animatediff_mode_without_vae(tmp_path, monkeypatch):
    # The checkpoint's own VAE is used, not the video VAE
    command = _video_command(
        tmp_path, monkeypatch, in_video_mode=1, in_animatediff_vae=None
    )
    assert '--vae' not in command


@pytest.mark.parametrize('mode', [0, 0.0, None])
def test_unet_mode(tmp_path, monkeypatch, mode):
    # AnimateDiff models stay selected in their tab but aren't used
    command = _video_command(tmp_path, monkeypatch, in_video_mode=mode)
    assert '--motion-module' not in command
    assert '--model' not in command
    assert os.path.basename(_flag(command, '--diffusion-model')) == (
        'wan2.1_t2v.gguf'
    )
    assert os.path.basename(_flag(command, '--vae')) == 'wan_vae.safetensors'
    assert os.path.basename(
        _flag(command, '--high-noise-diffusion-model')
    ) == 'wan2.2_high_noise.gguf'

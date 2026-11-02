"""Wan2.2 S2V: audio encoder and driving audio in the any2video command."""
import os


def _flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


def _video_command(tmp_path, monkeypatch, **params):
    from modules.core.cli.sdcpp_cli import Any2VideoRunner
    base = {
        'in_pprompt': 'a person is talking',
        'in_nprompt': '',
        'in_steps': 20,
        'in_width': 832,
        'in_height': 480,
        'in_cfg': 6.0,
        'in_seed': 42,
        'in_frames': 81,
        'in_fps': 16,
        'in_unet_model': 'wan2.2_s2v-14B-Q8_0.gguf',
        'in_unet_vae': 'wan_2.1_vae.safetensors',
        'in_umt5_xxl': 'umt5-xxl-encoder-Q8_0.gguf',
        'in_audio_encoder': 'wav2vec2_large_english_fp16.safetensors',
        'in_img_inp': str(tmp_path / 'portrait.png'),
        'in_s2v_audio': str(tmp_path / 'speech.wav'),
    }
    base.update(params)
    monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
    runner = Any2VideoRunner(mode='vid_gen', params=base)
    runner.build_command()
    return runner.command


def test_s2v_command(tmp_path, monkeypatch):
    command = _video_command(tmp_path, monkeypatch)
    assert os.path.basename(_flag(command, '--audio-encoder')) == (
        'wav2vec2_large_english_fp16.safetensors'
    )
    assert os.path.basename(_flag(command, '--audio')) == 'speech.wav'
    assert os.path.basename(_flag(command, '--init-img')) == 'portrait.png'


def test_no_audio_without_inputs(tmp_path, monkeypatch):
    command = _video_command(
        tmp_path, monkeypatch, in_audio_encoder=None, in_s2v_audio=None
    )
    assert '--audio-encoder' not in command
    assert '--audio' not in command


def test_animatediff_mode_drops_audio_encoder(tmp_path, monkeypatch):
    command = _video_command(
        tmp_path, monkeypatch, in_video_mode=1,
        in_motion_module='mm_sd15_v3.safetensors',
        in_animatediff_ckpt='realisticVisionV60B1.safetensors',
    )
    assert '--audio-encoder' not in command

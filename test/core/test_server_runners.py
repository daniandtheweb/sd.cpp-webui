import os

import pytest

DIR_KEYS = ("txt2img_dir", "img2img_dir", "imgedit_dir")


def _flags(command):
    """Parses a command list into a {flag: value} map."""
    flags = {}
    i = 0
    while i < len(command):
        token = command[i]
        if token.startswith('-'):
            if (i + 1 < len(command)
                    and not command[i + 1].startswith('-')):
                flags[token] = command[i + 1]
                i += 2
            else:
                flags[token] = True
                i += 1
        else:
            i += 1
    return flags


def _expected_path(value, exe_dir):
    """Mimics _make_relative: relative inside exe_dir, else absolute."""
    if not value or not os.path.isabs(str(value)):
        return value
    rel = os.path.relpath(str(value), start=exe_dir)
    return value if rel.startswith('..') else rel


@pytest.fixture(autouse=True)
def exe_name_mock(mocker):
    return mocker.patch(
        "modules.utils.sd_interface.exe_name", return_value="sd-cli"
    )


def test_api_runners_sequential_output_paths(app_root):
    from modules.shared_instance import config
    from modules.core.server import sdcpp_server

    dir_paths = {}
    for name in ("txt2img", "img2img", "imgedit"):
        dir_path = app_root / f"api_{name}"
        dir_path.mkdir(exist_ok=True)
        dir_paths[name] = dir_path

    saved = {key: config.data.get(key) for key in DIR_KEYS}

    try:
        config.update_settings(
            {
                "txt2img_dir": str(dir_paths["txt2img"]),
                "img2img_dir": str(dir_paths["img2img"]),
                "imgedit_dir": str(dir_paths["imgedit"]),
                "def_output_scheme": "Sequential",
            }
        )

        seed_files = {
            "txt2img": "5.png",
            "img2img": "10.png",
            "imgedit": "3.png",
        }
        for name, filename in seed_files.items():
            (dir_paths[name] / filename).touch()

        params = {"in_ip": "127.0.0.1", "in_port": "7860"}

        txt2img_runner = sdcpp_server.Txt2ImgApiRunner(params)
        txt2img_runner.prepare()

        img2img_runner = sdcpp_server.Img2ImgApiRunner(params)
        img2img_runner.prepare()

        imgedit_runner = sdcpp_server.ImgEditApiRunner(params)
        imgedit_runner.prepare()

        assert txt2img_runner.output_path == str(
            dir_paths["txt2img"] / "6.png"
        )
        assert img2img_runner.output_path == str(
            dir_paths["img2img"] / "11.png"
        )
        assert imgedit_runner.output_path == str(
            dir_paths["imgedit"] / "4.png"
        )
    finally:
        config.update_settings(saved)


def test_server_startup_command(app_root):
    from modules.shared_instance import config, SD_CLI, SD_SERVER
    from modules.core.server.manager import ServerRunner

    exe_dir = os.path.dirname(os.path.abspath(SD_CLI))

    params = {
        'ip': '127.0.0.1',
        'port': 1234,
        'in_diffusion_mode': 0,
        'f_ckpt_model': os.path.join(exe_dir, 'models', 'ckpt.gguf'),
        'f_ckpt_vae': os.path.join(exe_dir, 'models', 'vae.gguf'),
        'in_model_type': 'Q8_0',
        'in_tensor_type_rules': '0-6=f16',
        'in_threads': 8,
        'in_cache_bool': True,
        'in_cache_mode': 'easycache',
        'f_taesd': os.path.join(exe_dir, 'models', 'taesd.safetensors'),
        'in_phtmkr_bool': True,
        'f_phtmkr': os.path.join(exe_dir, 'models', 'pm.gguf'),
        'in_phtmkr_id': os.path.join(exe_dir, 'models', 'pm_id'),
        'in_phtmkr_emb': os.path.join(exe_dir, 'models', 'pm_emb.bin'),
        'in_upscl_bool': True,
        'f_upscl': os.path.join(exe_dir, 'models', 'up.gguf'),
        'in_cnnet_bool': True,
        'f_cnnet': os.path.join(exe_dir, 'models', 'cn.gguf'),
        'in_max_vram_table': [['*', '8']],
        'in_backend_table': [['*', 'vulkan0']],
        'in_params_backend_table': [['*', 'vulkan0']],
        'in_split_modes': [['diffusion', 'row']],
        'in_rpc_servers': '127.0.0.1:5000',
        'in_conditioning_cache_size': 16,
        'in_log_level': 'debug',
        'in_compression_quality': 95,
        'in_predict': 'epsilon',
    }

    runner = ServerRunner(params)
    runner.build_command()
    flags = _flags(runner.command)

    # Binary and network
    assert runner.command[0] == SD_SERVER
    assert flags['--listen-ip'] == '127.0.0.1'
    assert flags['--listen-port'] == '1234'

    # Model files (relativized)
    assert flags['--model'] == 'models/ckpt.gguf'
    assert flags['--vae'] == 'models/vae.gguf'

    # Quantization
    assert flags['--type'] == 'Q8_0'
    assert flags['--tensor-type-rules'] == '0-6=f16'

    # Components
    assert flags['-t'] == '8'
    assert flags['--cache-mode'] == 'easycache'
    assert flags['--taesd'] == 'models/taesd.safetensors'
    assert flags['--photo-maker'] == 'models/pm.gguf'
    assert flags['--pm-id-images-dir'] == 'models/pm_id'
    assert flags['--pm-id-embed-path'] == 'models/pm_emb.bin'
    assert flags['--upscale-model'] == 'models/up.gguf'
    assert flags['--control-net'] == 'models/cn.gguf'
    assert flags['--prediction'] == 'epsilon'

    # Directories (relativized when inside the exe directory)
    assert flags['--embd-dir'] == _expected_path(
        config.get('emb_dir'), exe_dir
    )
    assert flags['--lora-model-dir'] == _expected_path(
        config.get('lora_dir'), exe_dir
    )

    # Performance
    assert flags['--max-vram'] == '8'
    assert flags['--backend'] == 'vulkan0'
    assert flags['--params-backend'] == '*=vulkan0'
    assert flags['--split-mode'] == 'diffusion=row'

    # Common options
    assert flags['--rpc-servers'] == '127.0.0.1:5000'
    assert flags['--conditioning-cache-size'] == '16'
    assert flags['--log-level'] == 'debug'
    assert flags['--compression-quality'] == '95'
    assert '--auto-fit' not in flags


def test_server_startup_command_defaults_omitted(app_root):
    from modules.core.server.manager import ServerRunner

    runner = ServerRunner({'ip': '127.0.0.1', 'port': 1234})
    runner.build_command()
    flags = _flags(runner.command)

    omitted = (
        '--model', '--type', '--tensor-type-rules', '-t',
        '--cache-mode', '--taesd', '--photo-maker',
        '--pm-id-images-dir', '--pm-id-embed-path',
        '--upscale-model', '--control-net', '--prediction',
        '--max-vram', '--backend', '--params-backend',
        '--split-mode', '--rpc-servers',
    )
    for flag in omitted:
        assert flag not in flags

    # Unconditional directories remain
    assert '--embd-dir' in flags
    assert '--lora-model-dir' in flags

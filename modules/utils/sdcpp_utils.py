"""sd.cpp-webui - sdcpp.py utility module"""

import datetime
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from modules.gallery import get_next_media
from modules.shared_instance import SD_CLI

DEVICES_CACHE_PATH = os.path.join('user_data', 'devices_cache.json')


def list_devices(timeout: int = 10) -> List[str]:
    """
    Runs `sd-cli --list-devices` and returns the detected device names
    (one per 'name<TAB>description' line). Returns [] on any failure.
    """
    try:
        result = subprocess.run(
            [SD_CLI, '--list-devices'],
            capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.SubprocessError, OSError):
        return []

    devices = []
    for line in (result.stdout or '').splitlines():
        if '\t' not in line:
            continue
        name = line.split('\t', 1)[0].strip()
        if name:
            devices.append(name)
    return devices


def restart_server():
    """
    Restarts the sdcpp-webui.
    """
    print("\nRestarting server...")
    os.environ['SDCPP_IS_RESTART'] = 'true'
    new_args = [arg for arg in sys.argv if arg != '--autostart']
    os.execv(sys.executable, [sys.executable] + new_args)


def get_cached_devices() -> List[str]:
    """Returns the cached device list (empty list if no valid cache)."""
    try:
        with open(DEVICES_CACHE_PATH, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(d) for d in data]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def refresh_device_cache() -> List[str]:
    """
    Detects devices and overwrites the cache only on a successful
    detection (a failed run keeps the previous cache intact).
    """
    devices = list_devices()
    if devices:
        os.makedirs(os.path.dirname(DEVICES_CACHE_PATH), exist_ok=True)
        with open(DEVICES_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(devices, f)
    return devices


def extract_env_vars(params: Dict[str, Any]) -> Dict[str, str]:
    """
    Parses the params dictionary to find and extract environment
    variables, applying conditional logic as needed.
    """
    env_vars = {}

    is_vk_override_true = params.pop('env_vk_visible_override', False)
    vk_device_id = params.pop('env_GGML_VK_VISIBLE_DEVICES', None)
    is_cuda_override_true = params.pop('env_cuda_visible_override', False)
    cuda_device_id = params.pop('env_CUDA_VISIBLE_DEVICES', None)

    # Boolean "disable" flags: emitting them as 0 is a no-op, so only
    # include them when enabled (True).
    only_if_true = {
        'GGML_VK_DISABLE_COOPMAT',
        'GGML_VK_DISABLE_INTEGER_DOT_PRODUCT',
    }

    for key in list(params.keys()):
        if key.startswith("env_"):
            env_key = key[4:]
            value = params.pop(key)
            if env_key not in env_vars:
                if env_key in only_if_true and not value:
                    continue
                env_vars[env_key] = value

    if is_vk_override_true and vk_device_id is not None:
        env_vars['GGML_VK_VISIBLE_DEVICES'] = vk_device_id
    if is_cuda_override_true and cuda_device_id is not None:
        env_vars['CUDA_VISIBLE_DEVICES'] = cuda_device_id

    return env_vars


def generate_output_filename(
    directory: str, scheme: str, extension: str,
    name_parts: list, subctrl_id: int = 0
) -> str:
    """
    Generates a full output path based
    on the selected naming scheme.
    """

    prefix_str = ""

    match scheme:
        case "Timestamp":
            prefix_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        case "TimestampMS":
            prefix_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        case "EpochTime":
            prefix_str = str(int(time.time()))
        case "Sequential" | _:
            next_img = get_next_media(subctrl=subctrl_id)
            prefix_str = os.path.splitext(next_img)[0]

    if name_parts:
        suffix_str = "_".join(name_parts)
    else:
        suffix_str = ""

    if suffix_str:
        base_filename = f"{prefix_str}_{suffix_str}"
    else:
        base_filename = prefix_str

    test_path = os.path.join(directory, f"{base_filename}.{extension}")

    counter = 1
    while os.path.exists(test_path):
        test_path = os.path.join(
            directory, f"{base_filename}_{counter}.{extension}"
        )
        counter += 1

    return test_path


def build_device_choices() -> List[str]:
    """
    Returns the device dropdown choices: the 'default' sentinel first,
    followed by the cached detected devices.
    """
    return ['default'] + [d for d in get_cached_devices() if d != 'default']

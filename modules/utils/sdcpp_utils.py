"""sd.cpp-webui - sdcpp.py utility module"""

import os
import time
import datetime
from typing import Dict, Any

from modules.gallery import get_next_img


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

    for key in list(params.keys()):
        if key.startswith("env_"):
            env_key = key[4:]
            value = params.pop(key)
            if env_key not in env_vars:
                env_vars[env_key] = str(value)

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
            next_img = get_next_img(subctrl=subctrl_id)
            prefix_str = os.path.splitext(next_img)[0]

    if name_parts:
        suffix_str = "_".join(name_parts)
    else:
        suffix_str = ""

    if suffix_str:
        final_filename = f"{prefix_str}_{suffix_str}"
    else:
        final_filename = prefix_str

    return os.path.join(directory, f"{final_filename}.{extension}")

"""sd.cpp-webui - sdcpp.py utility module"""

from typing import Dict, Any


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
                env_vars[env_key] = value

    if is_vk_override_true and vk_device_id is not None:
        env_vars['GGML_VK_VISIBLE_DEVICES'] = vk_device_id
    if is_cuda_override_true and cuda_device_id is not None:
        env_vars['CUDA_VISIBLE_DEVICES'] = cuda_device_id

    return env_vars

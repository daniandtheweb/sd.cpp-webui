from modules.core.common.sd_common import CommonRunner
from modules.utils.sdcpp_utils import extract_env_vars


def test_extract_env_vars_preserves_value_types():
    """
    extract_env_vars must keep the original value types (bool / number)
    so _build_process_env can serialize them correctly.
    """
    params = {
        'env_GGML_VK_DISABLE_COOPMAT': True,
        'env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': False,
        # override flag + device id
        'env_vk_visible_override': True,
        'env_GGML_VK_VISIBLE_DEVICES': 0.0,
    }

    env = extract_env_vars(params)

    # The integer-dot flag is False, so it must be omitted (only-if-true).
    assert env == {
        'GGML_VK_DISABLE_COOPMAT': True,
        'GGML_VK_VISIBLE_DEVICES': 0.0,
    }


def test_extract_env_vars_omits_device_when_override_off():
    """Device ids must only be emitted when their override is enabled."""
    params = {
        'env_vk_visible_override': False,
        'env_GGML_VK_VISIBLE_DEVICES': 3.0,
        'env_cuda_visible_override': False,
        'env_CUDA_VISIBLE_DEVICES': 3.0,
    }

    env = extract_env_vars(params)

    assert 'GGML_VK_VISIBLE_DEVICES' not in env
    assert 'CUDA_VISIBLE_DEVICES' not in env


def test_build_process_env_injects_env_vars():
    """
    Regression: env vars must actually end up in the subprocess env.
    Previously every value was stringified upstream and only bool/int
    values were injected, so nothing was ever set.
    """
    params = {
        'env_GGML_VK_DISABLE_COOPMAT': True,
        'env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': False,
        'env_vk_visible_override': True,
        'env_GGML_VK_VISIBLE_DEVICES': 1.0,
        'env_cuda_visible_override': True,
        'env_CUDA_VISIBLE_DEVICES': 2.0,
    }

    runner = CommonRunner(params)
    env = runner._build_process_env()

    assert env['GGML_VK_DISABLE_COOPMAT'] == '1'
    # The integer-dot flag is False, so it must not be set at all.
    assert 'GGML_VK_DISABLE_INTEGER_DOT_PRODUCT' not in env
    assert env['GGML_VK_VISIBLE_DEVICES'] == '1'
    assert env['CUDA_VISIBLE_DEVICES'] == '2'

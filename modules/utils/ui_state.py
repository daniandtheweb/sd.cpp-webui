"""sd.cpp-webui - utils - UI state module"""


_SESSION_CACHE = {}


def get_session_value(option_key):
    """
    Helper to get the current model value.
    Prioritizes the live cache; falls back to config.
    """
    from modules.shared_instance import config

    if option_key and option_key in _SESSION_CACHE:
        return _SESSION_CACHE[option_key]

    return config.get(option_key)


def update_session_cache(key, value):
    """
    Public helper to allow other UI components (like checkboxes)
    to write to the global session cache.
    """
    _SESSION_CACHE[key] = value


class ModelState:
    """Class to manage the state of model parameters for the application.

    Attributes:
        bak_ckpt_model: The backup checkpoint model.
        bak_unet_model: The backup UNET model.
        bak_ckpt_vae: The backup checkpoint VAE model.
        bak_unet_vae: The backup UNET VAE model.
        bak_clip_g: The backup CLIP_G model.
        bak_clip_l: The backup CLIP_L model.
        bak_t5xxl: The backup T5-XXL model.
        bak_llm: The backup LLM model.
    """
    @property
    def bak_ckpt_model(self): return get_session_value('def_ckpt')

    @property
    def bak_unet_model(self): return get_session_value('def_unet')

    @property
    def bak_ckpt_vae(self): return get_session_value('def_ckpt_vae')

    @property
    def bak_unet_vae(self): return get_session_value('def_unet_vae')

    @property
    def bak_clip_g(self): return get_session_value('def_clip_g')

    @property
    def bak_clip_l(self): return get_session_value('def_clip_l')

    @property
    def bak_t5xxl(self): return get_session_value('def_t5xxl')

    @property
    def bak_llm(self): return get_session_value('def_llm')

    @property
    def bak_guidance_bool(self):
        return get_session_value('def_guidance_bool') is True

    @property
    def bak_flow_shift_bool(self):
        return get_session_value('def_flow_shift_bool') is True

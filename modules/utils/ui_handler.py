"""sd.cpp-webui - UI handler module"""

import gradio as gr

from modules.shared_instance import sd_options
from modules.ui.models import get_session_value


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

    def __init__(self):
        """Initializes using the LIVE session values."""
        self.bak_guidance_bool = False
        self.bak_flow_shift_bool = False

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

    def update(self, **kwargs):
        """Generic method to update state variables (mostly for booleans now)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def bak_ckpt_tab(self, *args, **kwargs):
        """
        Updates state from the checkpoint tab.
        Since models are now live properties, this does nothing 
        unless you add non-model settings to the Checkpoint tab later.
        """
        pass

    def bak_unet_tab(self, unet_model, unet_vae, clip_g, clip_l, t5xxl,
                     llm, guidance_bool, flow_shift_bool):
        """
        Updates state from the UNET tab.
        We ignore the model arguments (unet_model, etc.) because we read 
        those from the live cache. We only need to save the booleans.
        """
        self.update(
            bak_guidance_bool=guidance_bool,
            bak_flow_shift_bool=flow_shift_bool,
        )


model_state = ModelState()


def unet_tab_switch(ckpt_model, ckpt_vae, guidance_bool, guidance,
                    flow_shift_bool, flow_shift):
    """Switches to the UNET tab"""
    model_state.bak_ckpt_tab(ckpt_model, ckpt_vae)

    return (
        gr.update(value=1),                                                # + UNET Tab
        gr.update(value=None),                                             # - Checkpoint Model
        gr.update(value=model_state.bak_unet_model),                       # + UNET Model
        gr.update(value=None),                                             # - Checkpoint VAE
        gr.update(value=model_state.bak_unet_vae),                         # + UNET VAE
        gr.update(value=model_state.bak_clip_g),                           # + clip_g
        gr.update(value=model_state.bak_clip_l),                           # + clip_l
        gr.update(value=model_state.bak_t5xxl),                            # + t5xxl
        gr.update(value=model_state.bak_llm),                              # + llm
        gr.update(value=model_state.bak_guidance_bool, visible=True),      # + guidance_bool
        gr.update(visible=True),                                           # + guidance
        gr.update(value=model_state.bak_flow_shift_bool, visible=True),    # + flow_shift_bool
        gr.update(visible=True),                                           # + flow_shift
    )


def ckpt_tab_switch(unet_model, unet_vae, clip_g, clip_l, t5xxl,
                    llm, guidance_bool, guidance,
                    flow_shift_bool, flow_shift):
    """Switches to the checkpoint tab"""
    model_state.bak_unet_tab(unet_model, unet_vae, clip_g, clip_l, t5xxl,
                             llm, guidance_bool, flow_shift_bool)

    return (
        gr.update(value=0),                             # + Checkpoint Tab
        gr.update(value=model_state.bak_ckpt_model),    # + Checkpoint Model
        gr.update(value=None),                          # - UNET Model
        gr.update(value=model_state.bak_ckpt_vae),      # + Checkpoint VAE
        gr.update(value=None),                          # - UNET VAE
        gr.update(value=None),                          # - clip_g
        gr.update(value=None),                          # - clip_l
        gr.update(value=None),                          # - t5xxl
        gr.update(value=None),                          # - llm
        gr.update(value=False, visible=False),          # - guidance_bool
        gr.update(visible=False),                       # - guidance
        gr.update(value=False, visible=False),          # - flow_shift_bool
        gr.update(visible=False),                       # - flow_shift
    )


def update_interactivity(count, checkbox_value):
    """
    Generates a specified number of gr.update objects to set interactivity.
    """
    is_interactive = bool(checkbox_value)

    if count == 1:
        return gr.update(interactive=is_interactive)

    return tuple(gr.update(interactive=is_interactive) for _ in range(count))


def refresh_all_options():
    sd_options.refresh()
    return [
        gr.update(choices=sd_options.get_opt("samplers")),
        gr.update(choices=sd_options.get_opt("schedulers")),
        gr.update(choices=["none"] + sd_options.get_opt("previews")),
        gr.update(choices=["Default"] + sd_options.get_opt("prediction"))
    ]

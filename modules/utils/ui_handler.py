"""sd.cpp-webui - UI handler module"""

import gradio as gr

from modules.shared_instance import config


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
        bak_qwen2vl: The backup QWEN2VL model.
    """

    def __init__(self):
        """Initializes the ModelState with default values from the
        configuration."""
        self.bak_ckpt_model = config.get('def_ckpt')
        self.bak_unet_model = config.get('def_unet')
        self.bak_ckpt_vae = config.get('def_ckpt_vae')
        self.bak_unet_vae = config.get('def_unet_vae')
        self.bak_clip_g = config.get('def_clip_g')
        self.bak_clip_l = config.get('def_clip_l')
        self.bak_t5xxl = config.get('def_t5xxl')
        self.bak_qwen2vl = config.get('def_qwen2vl')
        self.bak_guidance_bool = False
        self.bak_flow_shift_bool = False

    def update(self, **kwargs):
        """Generic method to update state variables.

        Args:
            kwargs: Key-value pairs of attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"{key} is not a valid attribute of ModelState."
                )

    def bak_ckpt_tab(self, ckpt_model, ckpt_vae):
        """Updates the state with values from the checkpoint tab."""
        self.update(
            bak_ckpt_model=ckpt_model,
            bak_ckpt_vae=ckpt_vae,
        )

    def bak_unet_tab(self, unet_model, unet_vae, clip_g, clip_l, t5xxl,
                     qwen2vl, guidance_bool, flow_shift_bool):
        """Updates the state with values from the UNET tab."""
        self.update(
            bak_unet_model=unet_model,
            bak_unet_vae=unet_vae,
            bak_clip_g=clip_g,
            bak_clip_l=clip_l,
            bak_t5xxl=t5xxl,
            bak_qwen2vl=qwen2vl,
            bak_guidance_bool=guidance_bool,
            bak_flow_shift_bool=flow_shift_bool,
        )


model_state = ModelState()


def unet_tab_switch(ckpt_model, ckpt_vae, guidance_bool, guidance,
                    flow_shift_bool, flow_shift):
    """Switches to the UNET tab"""
    model_state.bak_ckpt_tab(ckpt_model, ckpt_vae)

    return (
        gr.update(value=1),
        gr.update(value=None),
        gr.update(value=model_state.bak_unet_model),
        gr.update(value=None),
        gr.update(value=model_state.bak_unet_vae),
        gr.update(value=model_state.bak_clip_g),
        gr.update(value=model_state.bak_clip_l),
        gr.update(value=model_state.bak_t5xxl),
        gr.update(value=model_state.bak_qwen2vl),
        gr.update(value=model_state.bak_guidance_bool, visible=True),
        gr.update(visible=True),
        gr.update(value=model_state.bak_flow_shift_bool, visible=True),
        gr.update(visible=True),
    )


def ckpt_tab_switch(unet_model, unet_vae, clip_g, clip_l, t5xxl,
                    qwen2vl, guidance_bool, guidance,
                    flow_shift_bool, flow_shift):
    """Switches to the checkpoint tab"""
    model_state.bak_unet_tab(unet_model, unet_vae, clip_g, clip_l, t5xxl,
                             qwen2vl, guidance_bool, flow_shift_bool)

    return (
        gr.update(value=0),
        gr.update(value=model_state.bak_ckpt_model),
        gr.update(value=None),
        gr.update(value=model_state.bak_ckpt_vae),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=False, visible=False),
        gr.update(visible=False),
        gr.update(value=False, visible=False),
        gr.update(visible=False),
    )


def update_interactivity(count, checkbox_value):
    """
    Generates a specified number of gr.update objects to set interactivity.
    """
    is_interactive = bool(checkbox_value)

    if count == 1:
        return gr.update(interactive=is_interactive)

    return tuple(gr.update(interactive=is_interactive) for _ in range(count))

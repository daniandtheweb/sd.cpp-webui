"""sd.cpp-webui - UI component for the model selection"""

import os

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models,
    get_loras
)
from modules.utils.ui_state import (
    get_session_value, update_session_cache
)
from .constants import RELOAD_SYMBOL


def create_model_widget(
    label, dir_key, option_key, **kwargs
):
    """
    Universal widget with auto-save to session cache.
    """
    full_path = config.get(dir_key)

    if 'info' not in kwargs and full_path:
        folder_name = os.path.basename(os.path.normpath(full_path))
        kwargs['info'] = f"{folder_name} folder"

    current_value = get_session_value(option_key)

    if dir_key == 'lora_dir':
        choices = list(get_loras())
    else:
        choices = list(get_models(full_path)) if full_path else []

    with gr.Group():
        path_component_txt = gr.Textbox(value=full_path, visible=False)

        with gr.Row():
            dropdown = gr.Dropdown(
                label=label,
                choices=choices,
                scale=kwargs.pop('scale', 7),
                value=current_value,
                interactive=True,
                **kwargs
            )

            if option_key:
                dropdown.input(
                    fn=lambda x: update_session_cache(option_key, x),
                    inputs=[dropdown],
                    outputs=[]
                )

        with gr.Row():
            reload_btn = gr.Button(
                value=RELOAD_SYMBOL,
                scale=1
            )
            if dir_key == 'lora_dir':
                reload_btn.click(
                    fn=lambda _: gr.update(choices=list(get_loras())),
                    inputs=[path_component_txt],
                    outputs=[dropdown]
                )
            else:
                reload_btn.click(
                    fn=reload_models,
                    inputs=[path_component_txt],
                    outputs=[dropdown]
                )
            clear_btn = gr.ClearButton(
                dropdown,
                scale=1
            )
            if option_key:
                clear_btn.click(
                    fn=lambda: update_session_cache(option_key, None),
                    inputs=[],
                    outputs=[]
                )

    return dropdown


def create_ckpt_model_sel_ui():
    """Create the checkpoint model selection UI using a helper for clarity"""
    with gr.Row():
        gr.Markdown("Supports: SD1.x, SD2.x, SD-Turbo, SDXL, SDXL-Turbo, NitroFusion")
    with gr.Row():
        with gr.Column():
            ckpt_model = create_model_widget(
                label="Checkpoint Model",
                dir_key='ckpt_dir',
                option_key='def_ckpt',
            )
        with gr.Column():
            ckpt_vae = create_model_widget(
                label="Checkpoint VAE",
                dir_key='vae_dir',
                option_key='def_ckpt_vae',
            )

    return {
        'in_ckpt_model': ckpt_model,
        'in_ckpt_vae': ckpt_vae,
    }


def create_unet_model_sel_ui():
    """Create the UNET model selection UI using a helper for clarity"""
    with gr.Row():
        gr.Markdown("Supports: SD3, SD3.5, FLUX.1-Krea-dev, FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, Chroma, Qwen Image, Z-Image-Turbo")
    with gr.Row():
        with gr.Column():
            unet_model = create_model_widget(
                label="UNET Model",
                dir_key='unet_dir',
                option_key='def_unet',
            )
        with gr.Column():
            unet_vae = create_model_widget(
                label="UNET VAE",
                dir_key='vae_dir',
                option_key='def_unet_vae',
            )
    with gr.Row():
        with gr.Column():
            clip_g = create_model_widget(
                label="clip_g",
                dir_key='txt_enc_dir',
                option_key='def_clip_g',
            )
        with gr.Column():
            clip_l = create_model_widget(
                label="clip_l",
                dir_key='txt_enc_dir',
                option_key='def_clip_l',
            )
        with gr.Column():
            t5xxl = create_model_widget(
                label="t5xxl",
                dir_key='txt_enc_dir',
                option_key='def_t5xxl',
            )
        with gr.Column():
            llm = create_model_widget(
                label="llm",
                dir_key='txt_enc_dir',
                option_key='def_llm',
            )
        with gr.Column():
            llm_vision = create_model_widget(
                label="llm_vision",
                dir_key='txt_enc_dir',
                option_key='def_llm_vision',
            )

    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_g': clip_g,
        'in_clip_l': clip_l,
        'in_t5xxl': t5xxl,
        'in_llm': llm,
        'in_llm_vision': llm_vision
    }


def create_img_model_sel_ui():
    """Create the image model selection UI."""
    diffusion_mode = gr.Number(value=0, visible=False)
    model_inputs = {'in_diffusion_mode': diffusion_mode}

    with gr.Tabs():
        with gr.Tab("Checkpoint") as ckpt_tab:
            ckpt_inputs = create_ckpt_model_sel_ui()
            model_inputs.update(ckpt_inputs)

        with gr.Tab("UNET") as unet_tab:
            unet_inputs = create_unet_model_sel_ui()
            model_inputs.update(unet_inputs)

    return {
        'inputs': model_inputs,
        'components': {
            'ckpt_tab': ckpt_tab,
            'unet_tab': unet_tab,
        }
    }


def create_imgedit_model_sel_ui():
    """Create the image edit selection UI"""
    with gr.Row():
        gr.Markdown("Supports: FLUX.1-Kontext-dev, Qwen Image Edit, Qwen Image Edit 2509")
    with gr.Row():
        with gr.Column():
            unet_model = create_model_widget(
                label="UNET Model",
                dir_key='unet_dir',
                option_key='def_unet',
            )
        with gr.Column():
            unet_vae = create_model_widget(
                label="UNET VAE",
                dir_key='vae_dir',
                option_key='def_unet_vae',
            )
    with gr.Row():
        with gr.Column():
            clip_l = create_model_widget(
                label="clip_l",
                dir_key='txt_enc_dir',
                option_key='def_clip_l',
            )
        with gr.Column():
            t5xxl = create_model_widget(
                label="t5xxl",
                dir_key='txt_enc_dir',
                option_key='def_t5xxl',
            )
        with gr.Column():
            llm = create_model_widget(
                label="llm",
                dir_key='txt_enc_dir',
                option_key='def_llm',
            )
        with gr.Column():
            llm_vision = create_model_widget(
                label="llm_vision",
                dir_key='txt_enc_dir',
                option_key='def_llm_vision',
            )

    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_l': clip_l,
        'in_t5xxl': t5xxl,
        'in_llm': llm,
        'in_llm_vision': llm_vision
    }


def create_video_model_sel_ui():
    """Create the video model selection UI"""
    with gr.Row():
        gr.Markdown("Supports: Wan2.1, Wan2.2")
    with gr.Row():
        with gr.Column():
            unet_model = create_model_widget(
                label="UNET Model",
                dir_key='unet_dir',
                option_key='def_unet',
            )
        with gr.Column():
            unet_vae = create_model_widget(
                label="UNET VAE",
                dir_key='vae_dir',
                option_key='def_unet_vae',
            )
    with gr.Row():
        with gr.Column():
            clip_vision_h = create_model_widget(
                label="clip_vision_h",
                dir_key='txt_enc_dir',
                option_key='def_clip_vision_h',
            )
        with gr.Column():
            umt5_xxl = create_model_widget(
                label="umt5_xxl",
                dir_key='txt_enc_dir',
                option_key='def_umt5_xxl',
            )
    with gr.Row():
        with gr.Accordion(
            label="High Noise", open=False
        ):
            high_noise_model = create_model_widget(
                label="high_noise_model",
                dir_key='unet_dir',
                option_key=None,
            )

    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_vision_h': clip_vision_h,
        'in_umt5_xxl': umt5_xxl,
        'in_high_noise_model': high_noise_model
    }

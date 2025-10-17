"""sd.cpp-webui - UI component for the model selection"""

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from .constants import RELOAD_SYMBOL

def _create_model_dropdown_widget(label, choices_dir_key, default_value_key, reload_input_component):
    """
    Creates a standardized gr.Group containing a Dropdown, Reload button, and Clear button.

    Args:
        label (str): The label for the gr.Dropdown.
        choices_dir_key (str): The config key for the directory path to get model choices.
        default_value_key (str): The config key for the default model selection.
        reload_input_component (gr.Textbox): The hidden textbox holding the path for the reload button.

    Returns:
        tuple: A tuple containing the created (gr.Dropdown, gr.Button)
    """
    with gr.Group():
        with gr.Row():
            dropdown = gr.Dropdown(
                label=label,
                choices=get_models(config.get(choices_dir_key)),
                scale=7,
                value=config.get(default_value_key),
                interactive=True
            )
        with gr.Row():
            reload_btn = gr.Button(
                value=RELOAD_SYMBOL,
                scale=1
            )
            reload_btn.click(
                reload_models,
                inputs=[reload_input_component],
                outputs=[dropdown]
            )
            gr.ClearButton(
                dropdown,
                scale=1
            )
    return dropdown, reload_btn

def create_ckpt_model_sel_ui():
    """Create the checkpoint model selection UI using a helper for clarity"""
    ckpt_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)

    with gr.Row():
        with gr.Column():
            ckpt_model, reload_ckpt_btn = _create_model_dropdown_widget(
                label="Checkpoint Model",
                choices_dir_key='ckpt_dir',
                default_value_key='def_ckpt',
                reload_input_component=ckpt_dir_txt
            )
        with gr.Column():
            ckpt_vae, reload_vae_btn = _create_model_dropdown_widget(
                label="Checkpoint VAE",
                choices_dir_key='vae_dir',
                default_value_key='def_ckpt_vae',
                reload_input_component=vae_dir_txt
            )
    return {
        'in_ckpt_model': ckpt_model,
        'in_ckpt_vae': ckpt_vae,
    }


def create_unet_model_sel_ui():
    """Create the UNET model selection UI using a helper for clarity"""
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    clip_dir_txt = gr.Textbox(value=config.get('clip_dir'), visible=False)

    with gr.Row():
        with gr.Column():
            unet_model, reload_unet_btn = _create_model_dropdown_widget(
                label="UNET Model",
                choices_dir_key='unet_dir',
                default_value_key='def_unet',
                reload_input_component=unet_dir_txt
            )
        with gr.Column():
            unet_vae, reload_unet_vae_btn = _create_model_dropdown_widget(
                label="UNET VAE",
                choices_dir_key='vae_dir',
                default_value_key='def_unet_vae',
                reload_input_component=vae_dir_txt
            )
    with gr.Row():
        with gr.Column():
            clip_g, reload_clip_g_btn = _create_model_dropdown_widget(
                label="clip_g",
                choices_dir_key='clip_dir',
                default_value_key='def_clip_g',
                reload_input_component=clip_dir_txt
            )
        with gr.Column():
            clip_l, reload_clip_l_btn = _create_model_dropdown_widget(
                label="clip_l",
                choices_dir_key='clip_dir',
                default_value_key='def_clip_l',
                reload_input_component=clip_dir_txt
            )
        with gr.Column():
            t5xxl, reload_t5xxl_btn = _create_model_dropdown_widget(
                label="t5xxl",
                choices_dir_key='clip_dir',
                default_value_key='def_t5xxl',
                reload_input_component=clip_dir_txt
            )
        with gr.Column():
            qwen2vl, reload_qwen2vl_btn = _create_model_dropdown_widget(
                label="qwen2vl",
                choices_dir_key='clip_dir',
                default_value_key='def_qwen2vl',
                reload_input_component=clip_dir_txt
            )
        with gr.Column():
            qwen2vl_vision, reload_qwen2vl_vision_btn = _create_model_dropdown_widget(
                label="qwen2vl_vision",
                choices_dir_key='clip_dir',
                default_value_key='def_qwen2vl_vision',
                reload_input_component=clip_dir_txt
            )
    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_g': clip_g,
        'in_clip_l': clip_l,
        'in_t5xxl': t5xxl,
        'in_qwen2vl': qwen2vl,
        'in_qwen2vl_vision': qwen2vl_vision
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


def create_video_model_sel_ui():
    """Create the video model selection UI"""
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    clip_dir_txt = gr.Textbox(value=config.get('clip_dir'), visible=False)

    with gr.Row():
        with gr.Column():
            unet_model, reload_unet_btn = _create_model_dropdown_widget(
                label="UNET Model",
                choices_dir_key='unet_dir',
                default_value_key='def_unet',
                reload_input_component=unet_dir_txt
            )
        with gr.Column():
            unet_vae, reload_unet_vae_btn = _create_model_dropdown_widget(
                label="UNET VAE",
                choices_dir_key='vae_dir',
                default_value_key='def_unet_vae',
                reload_input_component=vae_dir_txt
            )
    with gr.Row():
        with gr.Column():
            clip_vision_h, reload_clip_vision_h_btn = _create_model_dropdown_widget(
                label="clip_vision_h",
                choices_dir_key='clip_dir',
                default_value_key='def_clip_vision_h',
                reload_input_component=clip_dir_txt
            )
        with gr.Column():
            umt5_xxl, reload_umt5_xxl_btn = _create_model_dropdown_widget(
                label="umt5_xxl",
                choices_dir_key='clip_dir',
                default_value_key='def_umt5_xxl',
                reload_input_component=clip_dir_txt
            )
    with gr.Row():
        with gr.Accordion(
            label="High Noise", open=False
        ):
            high_noise_model, reload_high_noise_model = _create_model_dropdown_widget(
                label="high_noise_model",
                choices_dir_key='unet_dir',
                default_value_key=None,
                reload_input_component=unet_dir_txt
            )

    # Return the dictionary with all UI components
    return {
        'in_unet_model': unet_model,
        'in_unet_vae': unet_vae,
        'in_clip_vision_h': clip_vision_h,
        'in_umt5_xxl': umt5_xxl,
        'in_high_noise_model': high_noise_model
    }

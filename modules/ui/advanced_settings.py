"""sd.cpp-webui - UI component for miscellaneous options"""

import os

import gradio as gr

from modules.shared_instance import config

from modules.ui.constants import PREDICTION, PREVIEW


def create_extras_ui():
    """Create the extras UI"""
    with gr.Accordion(
        label="Extra", open=False
    ):
        threads = gr.Number(
            label="Threads",
            minimum=0,
            maximum=os.cpu_count(),
            value=0
        )
        with gr.Group():
            offload_to_cpu = gr.Checkbox(
                label="Offload to CPU")
            vae_cpu = gr.Checkbox(label="VAE on CPU")
            clip_cpu = gr.Checkbox(label="CLIP on CPU")
        rng = gr.Dropdown(
            label="RNG",
            choices=["std_default", "cuda"],
            value="cuda"
        )
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict')
        )
        output = gr.Textbox(
            label="Output Name (optional)", value=""
        )
        with gr.Group():
            flash_attn = gr.Checkbox(
                label="Flash Attention", value=config.get('def_flash_attn')
            )
            diffusion_conv_direct = gr.Checkbox(
                label="Conv2D Direct for diffusion",
                value=config.get('def_diffusion_conv_direct')
            )
            vae_conv_direct = gr.Checkbox(
                label="Conv2D Direct for VAE",
                value=config.get('def_vae_conv_direct')
            )
            force_sdxl_vae_conv_scale = gr.Checkbox(
                label="Force conv scale on SDXL VAE",
                value=False
            )
        with gr.Group():
            preview_mode = gr.Dropdown(
                label="Preview mode",
                choices=PREVIEW,
                value=config.get('def_preview_mode')
            )
            preview_interval = gr.Number(
                label="Preview interval",
                value=config.get('def_preview_interval'),
                minimum=1,
                interactive=True
            )
            preview_taesd = gr.Checkbox(
                label="TAESD for preview only",
                value=config.get('def_preview_taesd')
            )
        color = gr.Checkbox(
            label="Color", value=True
        )
        verbose = gr.Checkbox(label="Verbose")

    # Return the dictionary with all UI components
    return {
        'in_threads': threads,
        'in_offload_to_cpu': offload_to_cpu,
        'in_vae_cpu': vae_cpu,
        'in_clip_cpu': clip_cpu,
        'in_rng': rng,
        'in_predict': predict,
        'in_output': output,
        'in_color': color,
        'in_flash_attn': flash_attn,
        'in_diffusion_conv_direct': diffusion_conv_direct,
        'in_vae_conv_direct': vae_conv_direct,
        'in_force_sdxl_vae_conv_scale': force_sdxl_vae_conv_scale,
        'in_preview_mode': preview_mode,
        'in_preview_interval': preview_interval,
        'in_preview_taesd': preview_taesd,
        'in_verbose': verbose
    }

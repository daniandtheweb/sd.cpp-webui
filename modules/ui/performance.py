"""sd.cpp-webui - UI component for performance options"""

import os

import gradio as gr

from modules.shared_instance import config


def create_performance_ui():
    """Create the performance UI"""
    with gr.Accordion(
        label="Performance", open=False
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

    return{
        'in_threads': threads,
        'in_offload_to_cpu': offload_to_cpu,
        'in_vae_cpu': vae_cpu,
        'in_clip_cpu': clip_cpu,
        'in_flash_attn': flash_attn,
        'in_diffusion_conv_direct': diffusion_conv_direct,
        'in_vae_conv_direct': vae_conv_direct,
        'in_force_sdxl_vae_conv_scale': force_sdxl_vae_conv_scale,
    }

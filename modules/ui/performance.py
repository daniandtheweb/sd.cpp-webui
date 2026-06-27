"""sd.cpp-webui - UI component for performance options"""

import os

import gradio as gr

from modules.shared_instance import config


def create_performance_ui():
    """Create the performance UI"""
    with gr.Accordion(
        label="Performance", open=False
    ):
        backend_table = gr.Dataframe(
                headers=["Component", "Device"],
                datatype=["str", "str"],
                value=config.get('def_backend_table'),
                interactive=True,
                row_count=(5, "dynamic"),
                column_count=(2, "fixed"),
                label="Backend Configuration",
                type="array"
            )
        params_backend_table = gr.Dataframe(
                headers=["Component", "Device"],
                datatype=["str", "str"],
                value=config.get('def_params_backend_table'),
                interactive=True,
                row_count=(5, "dynamic"),
                column_count=(2, "fixed"),
                label="Backend Configuration",
                type="array"
            )
        threads = gr.Number(
            label="Threads",
            minimum=0,
            maximum=os.cpu_count(),
            value=0
        )
        with gr.Group():
            offload_to_cpu = gr.Checkbox(
                label="Offload to CPU",
                value=config.get('def_offload_to_cpu')
            )
            max_vram = gr.Slider(
                label="Max VRAM budget (GiB) for segmented graph execution",
                value=config.get('def_max_vram'),
                maximum=64,
                minimum=-1,
                step=0.1
            )
            stream_layers = gr.Checkbox(
                label="Stream layers",
                value=config.get('def_stream_layers')
            )
            eager_load = gr.Checkbox(
                label="Eager load",
                value=config.get('def_eager_load')
            )

        with gr.Group():
            flash_attn = gr.Checkbox(
                label="Flash Attention",
                value=config.get('def_flash_attn')
            )
            diffusion_fa = gr.Checkbox(
                label="Flash Attention in the diffusion model only",
                value=config.get('def_diffusion_fa')
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
                value=config.get('def_force_sdxl_vae_conv_scale')
            )

    return {
        'in_threads': threads,
        'in_max_vram': max_vram,
        'in_offload_to_cpu': offload_to_cpu,
        'in_stream_layers': stream_layers,
        'in_eager_load': eager_load,
        'in_backend_table': backend_table,
        'in_params_backend_table': params_backend_table,
        'in_flash_attn': flash_attn,
        'in_diffusion_fa': diffusion_fa,
        'in_diffusion_conv_direct': diffusion_conv_direct,
        'in_vae_conv_direct': vae_conv_direct,
        'in_force_sdxl_vae_conv_scale': force_sdxl_vae_conv_scale,
    }

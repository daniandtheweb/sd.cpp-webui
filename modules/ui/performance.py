"""sd.cpp-webui - UI component for performance options"""

import os

import gradio as gr

from modules.shared_instance import config
from modules.ui.backend_devices import create_backend_devices_ui


def create_performance_ui():
    """
    Creates the 'Backend devices' accordion (device placement) followed
    by the 'Performance' accordion (execution settings only).
    """
    (backend_state, params_state, split_state,
     offload_to_cpu, max_vram_state,
     auto_fit) = create_backend_devices_ui()

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
            eager_load = gr.Checkbox(
                label="Eager load",
                value=config.get('def_eager_load')
            )
            rpc_servers = gr.Textbox(
                label="RPC servers (comma-separated host:port)",
                placeholder="localhost:50052,192.168.1.3:50052",
                value=config.get('def_rpc_servers')
            )
            conditioning_cache_size = gr.Number(
                label="Conditioning cache size (0 disables)",
                minimum=0,
                precision=0,
                value=config.get('def_conditioning_cache_size')
            )
            disable_prefetch = gr.Checkbox(
                label="Disable async next-segment weight prefetch",
                value=config.get('def_disable_prefetch')
            )
            disable_segmented_compute = gr.Checkbox(
                label="Disable segmented compute (monolithic graph)",
                value=config.get('def_disable_segmented_compute')
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
            sage_attn = gr.Checkbox(
                label="SageAttention (native CUDA, FA fallback)",
                value=config.get('def_sage_attn')
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
            linear_scale = gr.Number(
                label="Linear input scale override (0 = model default)",
                value=config.get('def_linear_scale'),
                precision=None
            )
            attn_scale = gr.Number(
                label="FA K/V scale override (0 = model default, requires FA)",
                value=config.get('def_attn_scale'),
                precision=None
            )

    return {
        'in_threads': threads,
        'in_max_vram_table': max_vram_state,
        'in_eager_load': eager_load,
        'in_offload_to_cpu': offload_to_cpu,
        'in_auto_fit': auto_fit,
        'in_rpc_servers': rpc_servers,
        'in_conditioning_cache_size': conditioning_cache_size,
        'in_disable_prefetch': disable_prefetch,
        'in_disable_segmented_compute': disable_segmented_compute,
        'in_backend_table': backend_state,
        'in_params_backend_table': params_state,
        'in_split_modes': split_state,
        'in_flash_attn': flash_attn,
        'in_diffusion_fa': diffusion_fa,
        'in_sage_attn': sage_attn,
        'in_diffusion_conv_direct': diffusion_conv_direct,
        'in_vae_conv_direct': vae_conv_direct,
        'in_force_sdxl_vae_conv_scale': force_sdxl_vae_conv_scale,
        'in_linear_scale': linear_scale,
        'in_attn_scale': attn_scale,
    }

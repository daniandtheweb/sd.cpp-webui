"""sd.cpp-webui - UI component for performance options"""

import os

import gradio as gr

from modules.shared_instance import config
from modules.ui.backend_devices import create_backend_devices_ui
from modules.ui.constants import CONV_DIRECT_MODES


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
            force_sdxl_vae_conv_scale = gr.Checkbox(
                label="Force conv scale on SDXL VAE",
                value=config.get('def_force_sdxl_vae_conv_scale')
            )
            linear_scale = gr.Number(
                label="Linear input scale override (0 = model default)",
                value=config.get('def_linear_scale'),
                precision=None,
                minimum=0
            )
            attn_scale = gr.Number(
                label="FA K/V scale override (0 = model default, requires FA)",
                value=config.get('def_attn_scale'),
                precision=None,
                minimum=0
            )

        with gr.Group():
            with gr.Row():
                conv_direct = gr.Dropdown(
                    label="Conv Direct",
                    choices=CONV_DIRECT_MODES,
                    value=config.get('def_conv_direct'),
                    info="auto probes each backend for the faster method"
                )
                conv_direct_diffusion = gr.Dropdown(
                    label="Conv Direct (diffusion)",
                    choices=CONV_DIRECT_MODES,
                    value=config.get('def_conv_direct_diffusion'),
                    info="auto follows Conv Direct"
                )
                conv_direct_vae = gr.Dropdown(
                    label="Conv Direct (VAE)",
                    choices=CONV_DIRECT_MODES,
                    value=config.get('def_conv_direct_vae'),
                    info="auto follows Conv Direct"
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
        'in_conv_direct': conv_direct,
        'in_conv_direct_diffusion': conv_direct_diffusion,
        'in_conv_direct_vae': conv_direct_vae,
        'in_force_sdxl_vae_conv_scale': force_sdxl_vae_conv_scale,
        'in_linear_scale': linear_scale,
        'in_attn_scale': attn_scale,
    }

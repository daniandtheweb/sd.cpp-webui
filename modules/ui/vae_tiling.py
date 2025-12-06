"""sd.cpp-webui - UI component for VAE tiling"""

from functools import partial

import gradio as gr

from modules.utils.ui_handler import update_interactivity
from modules.shared_instance import config


def create_vae_tiling_ui():
    """Create VAE tiling UI"""
    with gr.Accordion(
        label="VAE Tiling", open=False
    ):
        vae_tiling = gr.Checkbox(
            label="VAE Tiling",
            value=config.get('def_vae_tiling')
        )
        vae_tile_overlap = gr.Slider(
            label="VAE Tile overlap",
            minimum=0,
            maximum=1,
            step=0.01,
            value=config.get('def_vae_tile_overlap'),
            interactive=False
        )
        vae_tile_size = gr.Number(
            label="VAE Tile size",
            minimum=1,
            maximum=1024,
            step=1,
            value=config.get('def_vae_tile_size'),
            interactive=False
        )
        vae_relative_bool = gr.Checkbox(
            label="Enable VAE relative tile size",
            value=config.get('def_vae_relative_bool'),
            interactive=False
        )
        vae_relative_tile_size = gr.Number(
            label="VAE relative tile size",
            minimum=0,
            maximum=1024,
            step=0.01,
            value=config.get('def_vae_relative_tile_size'),
            interactive=False
        )

        vae_tile_comp = [
            vae_tile_overlap, vae_tile_size, vae_relative_bool
        ]

        vae_tiling.change(
            partial(update_interactivity, len(vae_tile_comp)),
            inputs=vae_tiling,
            outputs=vae_tile_comp
        )

        vae_relative_bool.change(
            partial(update_interactivity, 1),
            inputs=vae_relative_bool,
            outputs=vae_relative_tile_size
        )

    return {
        'in_vae_tiling': vae_tiling,
        'in_vae_tile_overlap': vae_tile_overlap,
        'in_vae_tile_size': vae_tile_size,
        'in_vae_relative_bool': vae_relative_bool,
        'in_vae_relative_tile_size': vae_relative_tile_size
    }

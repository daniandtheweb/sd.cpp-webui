"""sd.cpp-webui - UI components for the hires fix widget"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models
)
from modules.utils.ui_events import update_interactivity
from .constants import RELOAD_SYMBOL, BUILTIN_UPSCALERS


def reload_hires_upscalers(models_folder):
    """Refreshes the hires upscaler dropdown, keeping the built-in names"""
    return gr.update(
        choices=BUILTIN_UPSCALERS + get_models(models_folder)
    )


def create_hires_ui():
    """Create the HiRes Fix UI"""
    hires_upscalers_dir_txt = gr.Textbox(
        value=config.get('upscl_dir'), visible=False
    )

    with gr.Accordion(
        label="HiRes Fix", open=False
    ):
        with gr.Row():
            hires_bool = gr.Checkbox(
                label="Enable HiRes Fix", value=False
            )
        hires_upscaler = gr.Dropdown(
            label="Upscaler",
            choices=BUILTIN_UPSCALERS + get_models(
                config.get('upscl_dir')
            ),
            value=config.get('def_hires_upscaler'),
            allow_custom_value=True,
            interactive=False
        )
        with gr.Row():
            reload_hires_upscaler_btn = gr.Button(
                value=RELOAD_SYMBOL,
                interactive=False
            )
            clear_hires_upscaler_btn = gr.ClearButton(
                hires_upscaler,
                interactive=False
            )
        with gr.Row():
            hires_scale = gr.Number(
                label="Scale (when width and height are 0)",
                minimum=0.1,
                precision=1,
                step=0.1,
                value=config.get('def_hires_scale'),
                interactive=False
            )
            hires_width = gr.Number(
                label="Target width (0 uses scale)",
                minimum=0,
                precision=0,
                step=1,
                value=config.get('def_hires_width'),
                interactive=False
            )
            hires_height = gr.Number(
                label="Target height (0 uses scale)",
                minimum=0,
                precision=0,
                step=1,
                value=config.get('def_hires_height'),
                interactive=False
            )
        hires_steps = gr.Number(
            label="Second pass steps (0 reuses steps)",
            minimum=0,
            precision=1,
            value=config.get('def_hires_steps'),
            interactive=False
        )
        hires_denoising_strength = gr.Slider(
            label="Second pass denoising strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=config.get('def_hires_denoising_strength'),
            interactive=False
        )
        hires_sigmas = gr.Textbox(
            label="Custom sigmas (comma-separated)",
            value=config.get('def_hires_sigmas'),
            interactive=False
        )
        hires_upscale_tile_size = gr.Number(
            label="Tile size for model upscalers",
            minimum=1,
            maximum=4096,
            precision=0,
            value=config.get('def_hires_upscale_tile_size'),
            interactive=False
        )

    hires_comp = [
        hires_upscaler, reload_hires_upscaler_btn,
        clear_hires_upscaler_btn, hires_scale, hires_width,
        hires_height, hires_steps, hires_denoising_strength,
        hires_sigmas, hires_upscale_tile_size
    ]

    reload_hires_upscaler_btn.click(
        reload_hires_upscalers,
        inputs=[hires_upscalers_dir_txt],
        outputs=[hires_upscaler]
    )

    hires_bool.change(
        partial(update_interactivity, len(hires_comp)),
        inputs=hires_bool,
        outputs=hires_comp
    )

    return {
        'in_hires_bool': hires_bool,
        'in_hires_upscaler': hires_upscaler,
        'in_hires_scale': hires_scale,
        'in_hires_width': hires_width,
        'in_hires_height': hires_height,
        'in_hires_steps': hires_steps,
        'in_hires_denoising_strength': hires_denoising_strength,
        'in_hires_sigmas': hires_sigmas,
        'in_hires_upscale_tile_size': hires_upscale_tile_size,
    }

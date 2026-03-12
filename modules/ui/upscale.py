"""sd.cpp-webui - UI components for the upscale widget"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.ui_events import update_interactivity
from .constants import RELOAD_SYMBOL


def create_upscl_ui():
    """Create the upscale UI"""
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)

    with gr.Accordion(
        label="Upscale", open=False
    ):
        upscl_bool = gr.Checkbox(
            label="Enable Upscale", value=False
        )
        upscl = gr.Dropdown(
            label="Upscaler",
            choices=get_models(config.get('upscl_dir')),
            value="",
            allow_custom_value=True,
            interactive=False
        )
        with gr.Row():
            reload_upscl_btn = gr.Button(
                value=RELOAD_SYMBOL,
                interactive=False
            )
            clear_upscl_btn = gr.ClearButton(
                upscl,
                interactive=False)
        upscl_rep = gr.Slider(
            label="Upscaler repeats",
            minimum=1,
            maximum=5,
            value=1,
            step=1,
            interactive=False
        )

    upscl_comp = [upscl, reload_upscl_btn, clear_upscl_btn, upscl_rep]

    reload_upscl_btn.click(
        reload_models, inputs=[upscl_dir_txt], outputs=[upscl]
    )

    upscl_bool.change(
        partial(update_interactivity, len(upscl_comp)),
        inputs=upscl_bool,
        outputs=upscl_comp
    )

    return {
        'in_upscl_bool': upscl_bool,
        'in_upscl': upscl,
        'in_upscl_rep': upscl_rep
    }

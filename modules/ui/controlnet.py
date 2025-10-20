"""sd.cpp-webui - UI component for ControlNet"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.ui_handler import update_interactivity
from .constants import RELOAD_SYMBOL

def create_cnnet_ui():
    """Create the ControlNet UI"""
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    with gr.Accordion(
        label="ControlNet", open=False
    ):
        cnnet_bool = gr.Checkbox(
            label="Enable ControlNet", value=False
        )
        with gr.Group():
            cnnet = gr.Dropdown(
                label="ControlNet",
                choices=get_models(config.get('cnnet_dir')),
                value=None,
                interactive=False
            )
            with gr.Row():
                reload_cnnet_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=False)
                clear_cnnet_btn = gr.ClearButton(
                    cnnet,
                    interactive=False
                )
        control_img = gr.Image(
            sources="upload", type="filepath",
            interactive=False
        )
        control_strength = gr.Slider(
            label="ControlNet strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.9,
            interactive=False
        )
        cnnet_cpu = gr.Checkbox(
            label="ControlNet on CPU",
            interactive=False
        )
        canny = gr.Checkbox(
            label="Canny (edge detection)",
            interactive=False
        )

    cnnet_comp = [
        cnnet, reload_cnnet_btn, clear_cnnet_btn, control_img,
        control_strength, cnnet_cpu, canny
    ]

    reload_cnnet_btn.click(
        reload_models,
        inputs=[cnnet_dir_txt],
        outputs=[cnnet]
    )

    cnnet_bool.change(
        partial(update_interactivity, len(cnnet_comp)),
        inputs=cnnet_bool,
        outputs=cnnet_comp
    )

    # Return the dictionary with all UI components
    return {
        'in_cnnet_bool': cnnet_bool,
        'in_cnnet': cnnet,
        'in_control_img': control_img,
        'in_control_strength': control_strength,
        'in_cnnet_cpu': cnnet_cpu,
        'in_canny': canny
    }

"""sd.cpp-webui - UI component for eta"""

from functools import partial

import gradio as gr

from modules.utils.ui_handler import update_interactivity


def create_eta_ui():
    """Create ETA UI, for DDIM and TCD"""
    with gr.Accordion(
        label="ETA (for DDIM and TCD)", open=False
    ):
        with gr.Row():
            eta_bool = gr.Checkbox(
                label="Enable ETA",
                value=False
            )
        with gr.Row():
            eta = gr.Number(
                label="ETA",
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                interactive=False
            )
    eta_bool.change(
        partial(update_interactivity, 1),
        inputs=eta_bool,
        outputs=eta
    )

    return {
        'in_eta_bool': eta_bool,
        'in_eta': eta
    }


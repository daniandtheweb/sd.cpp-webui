"""sd.cpp-webui - UI component for timestep shift"""

from functools import partial

import gradio as gr

from modules.utils.ui_handler import update_interactivity

def create_timestep_shift_ui():
    """Create timeshift step UI, for NitroFusion models"""
    with gr.Accordion(
        label="Timestep shift", open=False
    ):
        with gr.Row():
            timestep_shift_bool = gr.Checkbox(
                label="Enable timestep shift",
                value=False
            )
        with gr.Row():
            timestep_shift = gr.Slider(
                label="Timestep shift",
                minimum=0,
                maximum=2000,
                value=0,
                step=1,
                interactive=False
            )
    timestep_shift_bool.change(
        partial(update_interactivity, 1),
        inputs=timestep_shift_bool,
        outputs=timestep_shift
    )
    return {
        'in_timestep_shift_bool': timestep_shift_bool,
        'in_timestep_shift': timestep_shift
    }

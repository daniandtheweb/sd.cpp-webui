"""sd.cpp-webui - UI components for the SLG widget"""

from functools import partial

import gradio as gr

from modules.utils.ui_events import update_interactivity


def create_slg_ui():
    """Create SLG specific UI"""
    with gr.Accordion(
        label="Skip Layer Guidance settings", open=False
    ):
        slg_bool = gr.Checkbox(
            label="Enable SLG",
            value=False
        )

        slg_scale = gr.Number(
            label="SLG Scale",
            value=0,
            minimum=0,
            maximum=10,
            step=0.1,
            interactive=False
        )

        skip_layer_start = gr.Number(
            label="SLG Enabling Point",
            value=0.01,
            minimum=0,
            maximum=10,
            step=0.01,
            interactive=False
        )

        skip_layer_end = gr.Number(
            label="SLG Disabling Point",
            value=0.2,
            minimum=0,
            maximum=10,
            step=0.01,
            interactive=False
        )

        skip_layers = gr.Textbox(
            label="Layers to Skip",
            value="[7,8,9]",
            placeholder="[7,8,9]",
            interactive=False
        )

        slg_comp = [
            slg_scale, skip_layer_start,
            skip_layer_end, skip_layers
        ]

        slg_bool.change(
            partial(update_interactivity, len(slg_comp)),
            inputs=slg_bool,
            outputs=slg_comp
        )

        return {
            'in_slg_bool': slg_bool,
            'in_slg_scale': slg_scale,
            'in_skip_layer_start': skip_layer_start,
            'in_skip_layer_end': skip_layer_end,
            'in_skip_layers': skip_layers
        }

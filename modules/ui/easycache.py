"""sd.cpp-webui - UI components for the EasyCache widget"""

import gradio as gr


def create_easycache_ui():
    """Create EasyCache specific UI"""
    with gr.Accordion(
        label="EasyCache", open=False
    ):
        easycache_bool = gr.Checkbox(
            label="Enable EasyCache",
            value=False
        )
        ec_threshold = gr.Number(
            label="Threshold",
            value=0.2,
            minimum=0,
            maximum=1,
            step=0.001,
            interactive=True,
        )
        ec_start = gr.Number(
            label="Start percent",
            value=0.15,
            minimum=0,
            maximum=1,
            step=0.01,
            interactive=True,
        )
        ec_end = gr.Number(
            label="End percent",
            value=0.95,
            minimum=0,
            maximum=1,
            step=0.01,
            interactive=True,
        )

    return {
        'in_easycache_bool': easycache_bool,
        'in_ec_threshold': ec_threshold,
        'in_ec_start': ec_start,
        'in_ec_end': ec_end
    }

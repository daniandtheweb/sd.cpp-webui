"""sd.cpp-webui - UI components for the EasyCache widget"""

import gradio as gr

from modules.shared_instance import config


def create_easycache_ui():
    """Create EasyCache specific UI"""
    with gr.Accordion(
        label="EasyCache", open=False
    ):
        easycache_bool = gr.Checkbox(
            label="Enable EasyCache",
            value=config.get('def_easycache_bool')
        )
        ec_threshold = gr.Number(
            label="Threshold",
            value=config.get('def_ec_threshold'),
            minimum=0,
            maximum=1,
            step=0.001,
            interactive=True,
        )
        ec_start = gr.Number(
            label="Start percent",
            value=config.get('def_ec_start'),
            minimum=0,
            maximum=1,
            step=0.01,
            interactive=True,
        )
        ec_end = gr.Number(
            label="End percent",
            value=config.get('def_ec_end'),
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

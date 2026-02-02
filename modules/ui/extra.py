"""sd.cpp-webui - UI component for miscellaneous options"""

import gradio as gr

from modules.shared_instance import config

from modules.ui.constants import (
    RNG, SAMPLER_RNG, PREDICTION
)


def create_extras_ui():
    """Create the extras UI"""
    with gr.Accordion(
        label="Extra", open=False
    ):
        sigmas = gr.Textbox(
            label="Sigmas",
            value="",
            placeholder="Overrides scheduler",
            interactive=True
        )
        rng = gr.Dropdown(
            label="RNG",
            choices=RNG,
            value=config.get('def_rng')
        )
        sampler_rng = gr.Dropdown(
            label="Sampler RNG",
            choices=SAMPLER_RNG,
            value=config.get('def_sampler_rng')
        )
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict')
        )
        lora_apply = gr.Dropdown(
            label="Lora apply mode",
            choices=["auto", "immediately", "at_runtime"],
            value=config.get('def_lora_apply')
        )
        output = gr.Textbox(
            label="Output Name (optional)",
            value=config.get('def_output')
        )
        color = gr.Checkbox(
            label="Color",
            value=config.get('def_color')
        )
        verbose = gr.Checkbox(
            label="Verbose",
            value=config.get('def_verbose')
        )

    return {
        'in_sigmas': sigmas,
        'in_rng': rng,
        'in_sampler_rng': sampler_rng,
        'in_predict': predict,
        'in_lora_apply': lora_apply,
        'in_output': output,
        'in_color': color,
        'in_verbose': verbose
    }

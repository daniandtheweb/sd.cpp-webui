"""sd.cpp-webui - UI component for miscellaneous options"""

import gradio as gr

from modules.shared_instance import config

from modules.ui.constants import(
    RNG, SAMPLER_RNG, PREDICTION
)


def create_extras_ui():
    """Create the extras UI"""
    with gr.Accordion(
        label="Extra", open=False
    ):
        rng = gr.Dropdown(
            label="RNG",
            choices=RNG,
            value="cuda"
        )
        sampler_rng = gr.Dropdown(
            label="Sampler RNG",
            choices=SAMPLER_RNG,
            value="cuda"
        )
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict')
        )
        lora_apply = gr.Dropdown(
            label="Lora apply mode",
            choices=["auto", "immediately", "at_runtime"],
            value="auto"
        )
        output = gr.Textbox(
            label="Output Name (optional)", value=""
        )

        color = gr.Checkbox(
            label="Color", value=True
        )
        verbose = gr.Checkbox(label="Verbose")

    # Return the dictionary with all UI components
    return {
        'in_rng': rng,
        'in_sampler_rng': sampler_rng,
        'in_predict': predict,
        'in_lora_apply': lora_apply,
        'in_output': output,
        'in_color': color,
        'in_verbose': verbose
    }

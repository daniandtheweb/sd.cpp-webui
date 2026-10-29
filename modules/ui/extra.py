"""sd.cpp-webui - UI component for miscellaneous options"""

import gradio as gr

from modules.shared_instance import config

from modules.ui.constants import (
    RNG, SAMPLER_RNG, PREDICTION, LOG_LEVEL
)


def create_extras_ui(generation: bool = True, launch: bool = True):
    """
    Create the extras UI
    generation: show the generation options (sigmas)
    launch: show the options only set at launch in server mode
    """
    with gr.Accordion(
        label="Extra", open=False
    ):
        sigmas = gr.Textbox(
            label="Sigmas",
            value="",
            placeholder="Overrides scheduler",
            interactive=True,
            visible=generation
        )
        rng = gr.Dropdown(
            label="RNG",
            choices=RNG,
            value=config.get('def_rng'),
            visible=launch
        )
        sampler_rng = gr.Dropdown(
            label="Sampler RNG",
            choices=SAMPLER_RNG,
            value=config.get('def_sampler_rng'),
            visible=launch
        )
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict'),
            visible=launch
        )
        lora_apply = gr.Dropdown(
            label="Lora apply mode",
            choices=["auto", "immediately", "at_runtime"],
            value=config.get('def_lora_apply'),
            visible=launch
        )
        mmap = gr.Checkbox(
            label="MMAP",
            value=config.get('def_mmap'),
            visible=launch
        )
        color = gr.Checkbox(
            label="Color",
            value=config.get('def_color'),
            visible=launch
        )
        log_level = gr.Dropdown(
            label="Log Level",
            choices=LOG_LEVEL,
            value=config.get('def_log_level'),
            visible=launch
        )

    return {
        'in_sigmas': sigmas,
        'in_rng': rng,
        'in_sampler_rng': sampler_rng,
        'in_predict': predict,
        'in_lora_apply': lora_apply,
        'in_mmap': mmap,
        'in_color': color,
        'in_log_level': log_level
    }

"""sd.cpp-webui - UI component for the main generation settings"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.utils.ui_handler import update_interactivity
from modules.utils.image_utils import switch_sizes
from .constants import QUANTS, SAMPLERS, SCHEDULERS, SWITCH_V_SYMBOL


def create_quant_ui():
    """Create the model type selection UI"""
    with gr.Accordion(
        label="Quantization", open=False
    ):
        with gr.Row():
            model_type = gr.Dropdown(
                label="Quantization type",
                choices=QUANTS,
                value=config.get('def_type'),
                interactive=True
            )
        with gr.Row():
            with gr.Accordion(
                label="Tensor type rules",
                open=False
            ):
                tensor_type_rules = gr.Textbox(
                    label="Weight type per tensor pattern",
                    value="",
                    placeholder="example: \"^vae\\.=f16,model\\.=q8_0\"",
                    interactive=True
                )
    return {
        'in_model_type': model_type,
        'in_tensor_type_rules': tensor_type_rules
    }


def create_generation_settings_ui(unet_mode: bool = False):
    """Create settings UI"""
    with gr.Row():
        with gr.Column():
            sampling = gr.Dropdown(
                label="Sampling method",
                choices=SAMPLERS,
                value=config.get('def_sampling'),
                interactive=True
            )
        with gr.Column():
            scheduler = gr.Dropdown(
                label="Scheduler",
                choices=SCHEDULERS,
                value=config.get('def_scheduler'),
                interactive=True
            )

    with gr.Row():
        with gr.Column():
            with gr.Group():
                width = gr.Slider(
                    label="Width",
                    minimum=64,
                    maximum=4096,
                    value=config.get('def_width'),
                    step=64
                )
                height = gr.Slider(
                    label="Height",
                    minimum=64,
                    maximum=4096,
                    value=config.get('def_height'),
                    step=64
                )
                switch_size = gr.Button(
                    value=SWITCH_V_SYMBOL, scale=1
                )
                switch_size.click(
                    switch_sizes,
                    inputs=[height,
                            width],
                    outputs=[height,
                             width]
                )
        with gr.Column():
            with gr.Row():
                steps = gr.Slider(
                    label="Steps",
                    minimum=1,
                    maximum=100,
                    value=config.get('def_steps'),
                    step=1
                )
            with gr.Row():
                cfg = gr.Slider(
                    label="CFG Scale",
                    minimum=1,
                    maximum=30,
                    value=config.get('def_cfg'),
                    step=0.1,
                    interactive=True
                )

    with gr.Row():
        flow_shift_bool = gr.Checkbox(
            label="Enable Flow Shift", value=False,
            visible=unet_mode
        )
        flow_shift = gr.Number(
            label="Flow Shift",
            minimum=1.0,
            maximum=12.0,
            value=config.get('def_flow_shift'),
            interactive=False,
            step=0.1,
            visible=unet_mode
        )

        flow_shift_comp = [flow_shift]

        flow_shift_bool.change(
            partial(update_interactivity, len(flow_shift_comp)),
            inputs=flow_shift_bool,
            outputs=flow_shift_comp
        )

    with gr.Row():
        guidance_bool = gr.Checkbox(
            label="Enable distilled guidance", value=False,
            visible=unet_mode
        )
        guidance = gr.Slider(
            label="Guidance",
            minimum=0,
            maximum=30,
            value=config.get('def_guidance'),
            step=0.1,
            interactive=False,
            visible=unet_mode
        )

        guidance_comp = [guidance]

        guidance_bool.change(
            partial(update_interactivity, len(guidance_comp)),
            inputs=guidance_bool,
            outputs=guidance_comp
        )

    # Return the dictionary with all UI components
    return {
        'in_sampling': sampling,
        'in_steps': steps,
        'in_scheduler': scheduler,
        'in_width': width,
        'in_height': height,
        'in_cfg': cfg,
        'in_flow_shift_bool': flow_shift_bool,
        'in_flow_shift': flow_shift,
        'in_guidance_bool': guidance_bool,
        'in_guidance': guidance
    }

def create_bottom_generation_settings_ui():
    """Create bottom settings UI"""
    with gr.Row():
        clip_skip = gr.Slider(
            label="CLIP skip",
            minimum=-1,
            maximum=12,
            value=config.get('def_clip_skip'),
            step=1
        )

    with gr.Row():
        batch_count = gr.Slider(
            label="Batch count",
            minimum=1,
            maximum=99,
            value=config.get('def_batch_count'),
            step=1
        )

    return{
        'in_clip_skip': clip_skip,
        'in_batch_count': batch_count
    }

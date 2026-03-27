"""sd.cpp-webui - UI component for the high noise generation settings"""

from functools import partial

import gradio as gr

from modules.utils.ui_events import update_interactivity
from modules.shared_instance import config
from modules.ui.constants import SAMPLERS


def create_high_noise_generation_settings_ui():
    """Create high noise settings UI"""
    high_noise_bool = gr.Checkbox(
        label="Enable High Noise Generation (Wan2.2)", value=False
    )
    with gr.Row():
        high_noise_steps = gr.Slider(
            label="High Noise Steps",
            minimum=1,
            value=8,
            step=1,
            interactive=False
        )
        high_noise_cfg = gr.Slider(
            label="High Noise CFG Scale",
            minimum=0.0,
            value=3.5,
            step=0.1,
            interactive=False
        )
    with gr.Row():
        high_noise_sampling = gr.Dropdown(
            label="High Noise Sampling Method",
            choices=SAMPLERS,
            value=config.get('def_sampler'),
            allow_custom_value=True,
            interactive=False
        )
        high_noise_img_cfg = gr.Number(
            label="High Noise Image CFG",
            minimum=0.0,
            value=3.5,
            step=0.1,
            interactive=False
        )
    with gr.Row():
        high_noise_guidance = gr.Number(
            label="High Noise Guidance",
            minimum=0.0,
            value=3.5,
            step=0.1,
            interactive=False
        )
        high_noise_slg_scale = gr.Number(
            label="High Noise SLG Scale",
            value=0.0,
            step=0.1,
            interactive=False
        )
    with gr.Row():
        high_noise_skip_layer_start = gr.Number(
            label="High Noise SLG Start",
            value=0.01,
            step=0.01,
            interactive=False
        )
        high_noise_skip_layer_end = gr.Number(
            label="High Noise SLG End",
            value=0.2,
            step=0.01,
            interactive=False
        )
    with gr.Row():
        high_noise_skip_layers = gr.Textbox(
            label="High Noise Skip Layers",
            value="[7,8,9]",
            placeholder="[7,8,9]",
            interactive=False
        )
        high_noise_eta = gr.Number(
            label="High Noise ETA",
            value=0.0,
            step=0.1,
            interactive=False
        )

    high_noise_comp = [
        high_noise_steps, high_noise_cfg, high_noise_sampling,
        high_noise_img_cfg, high_noise_guidance,
        high_noise_slg_scale, high_noise_skip_layer_start,
        high_noise_skip_layer_end, high_noise_skip_layers,
        high_noise_eta
    ]

    high_noise_bool.change(
        partial(update_interactivity, len(high_noise_comp)),
        inputs=high_noise_bool,
        outputs=high_noise_comp
    )

    return {
        'in_high_noise_bool': high_noise_bool,
        'in_high_noise_steps': high_noise_steps,
        'in_high_noise_cfg': high_noise_cfg,
        'in_high_noise_sampling': high_noise_sampling,
        'in_high_noise_img_cfg': high_noise_img_cfg,
        'in_high_noise_guidance': high_noise_guidance,
        'in_high_noise_slg_scale': high_noise_slg_scale,
        'in_high_noise_skip_layer_start': high_noise_skip_layer_start,
        'in_high_noise_skip_layer_end': high_noise_skip_layer_end,
        'in_high_noise_skip_layers': high_noise_skip_layers,
        'in_high_noise_eta': high_noise_eta,
    }

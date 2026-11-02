"""sd.cpp-webui - UI components for the Qwen Image Layered widget"""

from functools import partial

import gradio as gr

from modules.utils.ui_events import update_interactivity


def create_qwen_layered_ui():
    """Create Qwen Image Layered specific UI"""
    with gr.Accordion(
        label="Qwen Image Layered", open=False
    ):
        gr.Markdown(
            "Splits a single reference image into RGBA layers. Requires "
            "the Qwen Image Layered model, the Qwen 2.5 VL text encoder "
            "and the Layered VAE. Recommended: euler, flow shift 3, "
            "CFG 4, 20 steps, 640x640. The prompt is optional and isn't "
            "auto-captioned: describe the whole image, including hidden "
            "parts, not the individual layers."
        )
        qwen_layered_bool = gr.Checkbox(
            label="Enable Qwen Image Layered",
            value=False,
            info="Required with the Layered model, or its outputs "
                 "won't be shown"
        )
        qwen_image_layers = gr.Number(
            label="Layers (outputs layers + 1 images)",
            minimum=0,
            precision=0,
            step=1,
            value=3,
            interactive=False
        )

    qwen_layered_bool.change(
        partial(update_interactivity, 1),
        inputs=qwen_layered_bool,
        outputs=qwen_image_layers
    )

    return {
        'in_qwen_layered_bool': qwen_layered_bool,
        'in_qwen_image_layers': qwen_image_layers,
    }

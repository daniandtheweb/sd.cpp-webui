"""sd.cpp-webui - UI component for output options"""

import gradio as gr

from modules.shared_instance import config


def create_output_ui():
    """Create the output UI"""
    with gr.Accordion(
        label="Output", open=False
    ):
        output = gr.Textbox(
            label="Output Name (optional)",
            value=config.get('def_output')
        )
        disable_img_metadata = gr.Checkbox(
            label="Disable image metadata",
            value=config.get('def_disable_img_metadata')
        )
        compression_quality = gr.Slider(
            label="Compression quality",
            minimum=1,
            maximum=100,
            step=1,
            value=config.get('def_compression_quality')
        )

    return {
        'in_output': output,
        'in_disable_img_metadata': disable_img_metadata,
        'in_compression_quality': compression_quality
    }

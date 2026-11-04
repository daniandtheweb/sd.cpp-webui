"""sd.cpp-webui - UI components for the TAESD widget"""

import gradio as gr

from modules.loader import (
    get_models, reload_models
)
from modules.shared_instance import config
from .constants import RELOAD_SYMBOL


def create_taesd_ui():
    """Create TAESD specific UI"""
    # Directory Textbox
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)

    with gr.Accordion(
        label="TAESD", open=False
    ):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    with gr.Row():
                        taesd_model = gr.Dropdown(
                            label="TAESD",
                            choices=get_models(config.get('taesd_dir')),
                            value=config.get('def_taesd'),
                            allow_custom_value=True,
                            interactive=True
                        )
                    with gr.Row():
                        reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL)
                        gr.ClearButton(taesd_model)
        preview_taesd = gr.Checkbox(
            label="TAESD for preview only",
            info="The VAE still decodes the final image. "
                 "Needs a preview mode that uses TAESD",
            value=config.get('def_preview_taesd')
        )

    reload_taesd_btn.click(
        reload_models, inputs=[taesd_dir_txt], outputs=[taesd_model]
    )

    return {
        'in_taesd': taesd_model,
        'in_preview_taesd': preview_taesd
    }

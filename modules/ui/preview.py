"""sd.cpp-webui - UI component for preview options"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.ui.constants import PREVIEW
from modules.utils.ui_handler import update_interactivity


def create_preview_ui():
    with gr.Accordion(
        label="Preview", open=False
    ):
        preview_bool = gr.Checkbox(
            label="Enable Preview",
            value=config.get('def_preview_bool')
        )
        preview_mode = gr.Dropdown(
            label="Preview mode",
            choices=PREVIEW,
            value=config.get('def_preview_mode')
        )
        preview_interval = gr.Number(
            label="Preview interval",
            value=config.get('def_preview_interval'),
            minimum=1,
            interactive=True
        )
        preview_taesd = gr.Checkbox(
            label="TAESD for preview only",
            value=config.get('def_preview_taesd')
        )
        preview_noisy = gr.Checkbox(
            label="Preview noisy",
            value=config.get('def_preview_noisy')
        )

    preview_comp = [
        preview_mode, preview_interval, preview_taesd,
        preview_noisy
    ]

    preview_bool.change(
        partial(update_interactivity, len(preview_comp)),
        inputs=preview_bool,
        outputs=preview_comp
    )

    return{
        'in_preview_bool': preview_bool,
        'in_preview_mode': preview_mode,
        'in_preview_interval': preview_interval,
        'in_preview_taesd': preview_taesd,
        'in_preview_noisy': preview_noisy,
    }

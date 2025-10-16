"""sd.cpp-webui - UI component for the experimental widget"""

import gradio as gr

from .constants import PREVIEW


def create_experimental_ui():
    """Create experimental UI"""
    with gr.Accordion(
        label="Experimental", open=False
    ):
        preview_mode = gr.Dropdown(
            label="Preview mode (WIP: PR #522)",
            choices=PREVIEW,
            value="none"
        )
        preview_interval = gr.Number(
            label="Preview interval (PR #522)",
            value=1,
            minimum=1,
            interactive=True
        )
        preview_taesd = gr.Checkbox(
            label="TAESD for preview only (WIP: PR #522)"
        )
    return {
        'in_preview_mode': preview_mode,
        'in_preview_interval': preview_interval,
        'in_preview_taesd': preview_taesd
    }

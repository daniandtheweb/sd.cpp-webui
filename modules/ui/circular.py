"""sd.cpp-webui - UI components for the circular padding widget"""

import gradio as gr

from modules.ui.constants import CIRCULAR_PADDING


def create_circular_ui():
    """Create circular padding specific UI"""
    with gr.Accordion(
        label="Circular settings", open=False
    ):
        with gr.Row():
            circular_padding = gr.Dropdown(
                label="Circular padding",
                choices=CIRCULAR_PADDING,
                value=CIRCULAR_PADDING[0],
                interactive=True
            )

        return {
            'in_circular_padding': circular_padding,
        }

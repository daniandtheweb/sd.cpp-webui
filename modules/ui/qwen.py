"""sd.cpp-webui - UI components for the qwen widget"""

import gradio as gr


def create_qwen_ui():
    """Create Qwen specific UI"""
    with gr.Accordion(
        label="Qwen settings", open=False
    ):
        with gr.Row():
            enable_zero_cond_t = gr.Checkbox(
                label="Enable zero_cond_t for Qwen Image"
            )

        return {
            'in_enable_zero_cond_t': enable_zero_cond_t,
        }

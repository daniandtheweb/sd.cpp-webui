"""sd.cpp-webui - UI components for the chroma widget"""

import gradio as gr


def create_chroma_ui():
    """Create Chroma specific UI"""
    with gr.Accordion(
        label="Chroma settings", open=False
    ):
        with gr.Row():
            disable_dit_mask = gr.Checkbox(
                label="Disable DiT mask for Chroma",
            )
        with gr.Row():
            enable_t5_mask = gr.Checkbox(
                label="Enable T5 mask for Chroma",
            )
            t5_mask_pad = gr.Slider(
                label="T5 mask pad size for Chroma",
                minimum=0,
                maximum=1024,
                value=1,
                step=1,
            )
        return {
            'in_disable_dit_mask': disable_dit_mask,
            'in_enable_t5_mask': enable_t5_mask,
            'in_t5_mask_pad': t5_mask_pad
        }

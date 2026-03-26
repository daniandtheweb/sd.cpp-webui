"""sd.cpp-webui - UI components for the reference image settings widget"""

import gradio as gr


def create_ref_img_ui():
    """Create Reference Image settings specific UI"""
    with gr.Accordion(
        label="Reference Image settings", open=False
    ):
        with gr.Row():
            increase_ref_index = gr.Checkbox(
                label="Increase the reference image index automatically"
            )
            disable_auto_resize_ref_image = gr.Checkbox(
                label="Disable auto resize of reference images"
            )

        return {
            'in_increase_ref_index': increase_ref_index,
            'in_disable_auto_resize_ref_image': disable_auto_resize_ref_image,
        }

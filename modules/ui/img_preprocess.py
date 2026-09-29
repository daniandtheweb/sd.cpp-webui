"""sd.cpp-webui - UI component for image preprocessing"""

import gradio as gr

from modules.shared_instance import config


def create_img_preprocess_ui():
    """Create the image preprocessing UI"""
    with gr.Accordion(
        label="Image Preprocessing", open=False
    ):
        img_preprocess = gr.Textbox(
            label="Image preprocess rules (--image-preprocess)",
            value=config.get('def_img_preprocess'),
            placeholder="target=control,mode=fit-pad,"
                        "canny=true;target=mask,filter=nearest-exact"
        )

    return {
        'in_img_preprocess': img_preprocess
    }

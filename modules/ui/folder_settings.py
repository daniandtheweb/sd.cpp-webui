"""sd.cpp-webui - UI component for the folders in the options ui"""

import gradio as gr

from modules.shared_instance import config


FOLDER_CONFIG = {
    'ckpt_dir': "Checkpoint folder",
    'unet_dir': "UNET folder",
    'vae_dir': "VAE folder",
    'txt_enc_dir': "Text encoders folder",
    'emb_dir': "Embeddings folder",
    'lora_dir': "Lora folder",
    'taesd_dir': "TAESD folder",
    'phtmkr_dir': "PhotoMaker folder",
    'upscl_dir': "Upscaler folder",
    'cnnet_dir': "ControlNet folder",
    'txt2img_dir': "txt2img outputs folder",
    'img2img_dir': "img2img outputs folder",
    'any2video_dir': "any2video output folder",
}


def create_folders_opt_ui():
    """Create the folder options UI programmatically."""
    ui_components = {}
    with gr.Row():
        with gr.Accordion(label="Folders", open=False):
            for key, label in FOLDER_CONFIG.items():
                textbox = gr.Textbox(
                    label=label,
                    value=config.get(key),
                    interactive=True
                )
                ui_components[f'{key}_txt'] = textbox

    return ui_components

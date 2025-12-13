"""sd.cpp-webui - Model loader module"""

import os
from typing import List

import gradio as gr

from modules.shared_instance import config


SUPPORTED_EXTENSIONS = (".gguf", ".safetensors", ".sft", ".pth", ".ckpt")
MODEL_DIR_MAP = {
    "Checkpoint": config.get('ckpt_dir'),
    "UNET": config.get('unet_dir'),
    "VAE": config.get('vae_dir'),
    "clip_g": config.get('txt_enc_dir'),
    "clip_l": config.get('txt_enc_dir'),
    "t5xxl": config.get('txt_enc_dir'),
    "llm": config.get('txt_enc_dir'),
    "taesd": config.get('taesd_dir'),
    "Lora": config.get('lora_dir'),
    "Embeddings": config.get('emb_dir'),
    "Upscalers": config.get('upscl_dir'),
    "ControlNet": config.get('cnnet_dir')
}


def get_models(models_folder: str) -> List[str]:
    """
    Lists all supported models in a folder.

    Args:
        models_folder (str): The path to the directory to scan.

    Returns:
        List[str]: A list of model filenames or an empty list.
    """
    if not os.path.isdir(models_folder):
        print(f"The {models_folder} folder does not exist.")
        return []

    try:
        models = [
            model
            for model in os.listdir(models_folder)
            if (os.path.isfile(os.path.join(models_folder, model))
                and model.endswith(SUPPORTED_EXTENSIONS))
        ]
        return sorted(models)
    except OSError as e:
        print(f"Could not read files from '{models_folder}': {e}")
        return []


def reload_models(models_folder: str) -> gr.Dropdown:
    """
    Creates a Gradio update object to refresh a model dropdown.

    Args:
        models_folder (str): The directory containing the models to list.

    Returns:
        gr.Dropdown: A Gradio update object with the new list of models.
    """
    return gr.update(choices=get_models(models_folder))


def model_choice(model_type: str) -> gr.Textbox:
    """
    Creates a Gradio update object to set the value of a Textbox
    to the directory of the selected model type.

    Args:
        model_type (str): The type of model selected.

    Returns:
        gr.Textbox: A Gradio update object with the corresponding
                    directory path.
    """
    # Get the directory from the model_map based on the model_type
    model_dir = MODEL_DIR_MAP.get(model_type)

    if model_dir is None:
        print(f"Model type '{model_type}' not found in MODEL_DIR_MAP.")
        return gr.update(value="")

    return gr.update(value=model_dir)

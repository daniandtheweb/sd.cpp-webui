"""sd.cpp-webui - Model loader module"""

import os

import gradio as gr

from modules.config import (
    sd_dir, flux_dir, vae_dir, clip_l_dir, t5xxl_dir, taesd_dir,
    lora_dir, emb_dir, upscl_dir, cnnet_dir
)


# Dictionary to map model types to their corresponding directories
model_map = {
    "Stable-Diffusion": sd_dir,
    "FLUX": flux_dir,
    "VAE": vae_dir,
    "clip_l": clip_l_dir,
    "t5xxl": t5xxl_dir,
    "taesd": taesd_dir,
    "Lora": lora_dir,
    "Embeddings": emb_dir,
    "Upscalers": upscl_dir,
    "ControlNet": cnnet_dir
}


def get_models(models_folder):
    """Lists models in a folder"""
    if os.path.isdir(models_folder):
        extensions = (".gguf", ".safetensors", ".sft", ".pth", ".ckpt")
        models = [model for model in os.listdir(models_folder)
                  if os.path.isfile(os.path.join(models_folder, model)) and
                  model.endswith(extensions)]
        return models

    print(f"The {models_folder} folder does not exist.")
    return []


def reload_models(models_folder):
    """Reloads models list"""
    refreshed_models = gr.update(choices=get_models(models_folder))
    return refreshed_models


def model_choice(model_type):
    """Outputs the folder of the selected model type"""
    # Get the directory from the model_map based on the model_type
    model_dir = model_map.get(model_type)

    if model_dir is None:
        print(f"Model type '{model_type}' not recognized.")
        return gr.update(value="")

    model_dir_txt = gr.update(value=model_dir)
    return model_dir_txt

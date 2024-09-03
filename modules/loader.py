"""sd.cpp-webui - Model loader module"""

import os

import gradio as gr

from modules.config import (
    sd_dir, flux_dir, vae_dir, clip_l_dir, t5xxl_dir, taesd_dir,
    lora_dir, emb_dir, upscl_dir, cnnet_dir
)


model_dir = sd_dir


def get_models(models_folder, safety):
    """Lists models in a folder"""
    if os.path.isdir(models_folder):
        if safety == 1:
            extensions = (".gguf", ".safetensors", ".pth")
        else:
            extensions = (".gguf", ".safetensors", ".pth", ".ckpt")

        models = [model for model in os.listdir(models_folder)
                  if os.path.isfile(os.path.join(models_folder, model)) and
                  model.endswith(extensions)]
        return models

    print(f"The {models_folder} folder does not exist.")
    return []


def reload_models(models_folder, safety):
    """Reloads models list"""
    refreshed_models = gr.update(choices=get_models(models_folder, safety))
    return refreshed_models


def model_choice(model_type):
    """Outputs the folder of the selected model type"""
    global model_dir
    match model_type:
        case "Stable-Diffusion":
            model_dir = sd_dir
        case "FLUX":
            model_dir = flux_dir
        case "VAE":
            model_dir = vae_dir
        case "clip_l":
            model_dir = clip_l_dir
        case "t5xxl":
            model_dir = t5xxl_dir
        case "taesd":
            model_dir = taesd_dir
        case "Lora":
            model_dir = lora_dir
        case "Embeddings":
            model_dir = emb_dir
        case "Upscalers":
            model_dir = upscl_dir
        case "ControlNet":
            model_dir = cnnet_dir
    model_dir_txt = gr.update(value=model_dir)
    return model_dir_txt

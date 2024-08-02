"""sd.cpp-webui - Model loader module"""

import os

import gradio as gr

from modules.config import model_dir

def get_models(models_folder):
    """Lists models in a folder"""
    if os.path.isdir(models_folder):
        models = [model for model in os.listdir(models_folder)
                 if os.path.isfile(models_folder + model) and
                 (model.endswith((".gguf", ".safetensors", ".pth")))]
        return models

    print(f"The {models_folder} folder does not exist.")
    return []


def reload_models(models_folder):
    """Reloads models list"""
    refreshed_models = gr.update(choices=get_models(models_folder))
    return refreshed_models


def get_hf_models():
    """Lists convertible models in a folder"""
    fmodels_dir = model_dir
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(fmodels_dir + model) and
                (model.endswith((".safetensors", ".ckpt", ".pth", ".gguf")))]

    print(f"The {fmodels_dir} folder does not exist.")
    return []


def reload_hf_models():
    """Reloads convertible models list"""
    refreshed_models = gr.update(choices=get_hf_models())
    return refreshed_models

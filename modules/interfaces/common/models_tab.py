"""sd.cpp-webui - Models UI"""

import json
import os

import gradio as gr

from modules.ui.model_list import create_model_list
from modules.utils.model_files import (
    MODEL_TYPES, list_models, model_folder, remove_model, rename_model
)


def show_models(model_type, message="", ok=True):
    """The value of the list: the models of a type and a message."""
    folder = model_folder(model_type)
    if not message and not (folder and os.path.isdir(folder)):
        message, ok = f"The {model_type} folder does not exist.", False
    return json.dumps({
        'folder': folder or "",
        'models': list_models(model_type),
        'message': message,
        'ok': ok
    })


def change_model(model_type, evt: gr.EventData):
    """A button of the list was clicked: rename or remove that model."""
    action = getattr(evt, 'action', "")
    name = getattr(evt, 'name', "")
    if action == "rename":
        message, done = rename_model(
            model_type, name, getattr(evt, 'new_name', "")
        )
        return show_models(model_type, message, done is not None)
    if action == "remove":
        message, done = remove_model(model_type, name)
        return show_models(model_type, message, done)
    return show_models(model_type)


with gr.Blocks() as models_block:
    models_title = gr.Markdown("# Models")

    model_type = gr.Dropdown(
        label="Model Type",
        choices=list(MODEL_TYPES),
        value="Checkpoint",
        interactive=True
    )

    model_list = create_model_list(show_models("Checkpoint"))

    model_type.input(
        show_models,
        inputs=[model_type],
        outputs=[model_list]
    )

    model_list.click(
        change_model,
        inputs=[model_type],
        outputs=[model_list]
    )

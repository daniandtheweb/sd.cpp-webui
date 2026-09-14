"""sd.cpp-webui - UI component for the resolution presets feature"""

import gradio as gr

from modules.shared_instance import resolution_manager
from .constants import RELOAD_SYMBOL


def create_resolutions_ui(width, height):
    """
    Create the resolution presets UI.
    Binds all events internally; nothing here is a generation input.
    """

    def apply_resolution(name):
        if not name:
            gr.Warning("Please select a resolution to apply.")
            return gr.skip(), gr.skip()
        resolution = resolution_manager.get_resolution(name)
        if resolution is None:
            return gr.skip(), gr.skip()
        return gr.update(value=resolution[0]), gr.update(value=resolution[1])

    def save_and_refresh_resolution(name, res_width, res_height):
        if not name or not str(name).strip():
            gr.Warning("Please give a name to the resolution preset.")
            return gr.skip()
        clean_name = str(name).strip()
        if not resolution_manager.add_resolution(clean_name, res_width, res_height):
            gr.Warning(f"Resolution preset '{clean_name}' already exists.")
            return gr.skip()
        gr.Info(f"Resolution preset '{clean_name}' saved successfully.")
        return gr.update(choices=resolution_manager.get_resolutions(), value=clean_name)

    def delete_and_refresh_resolution(name):
        if not name:
            gr.Warning("Please select a resolution to delete.")
            return gr.skip()
        if not resolution_manager.delete_resolution(name):
            gr.Warning(f"Resolution preset '{name}' is built-in and cannot be deleted.")
            return gr.skip()
        gr.Info(f"Resolution preset '{name}' deleted.")
        return gr.update(choices=resolution_manager.get_resolutions(), value=None)

    def refresh_resolution_list():
        return gr.update(choices=resolution_manager.get_resolutions())

    with gr.Accordion(
        label="Saved resolutions", open=False
    ):
        with gr.Group():
            with gr.Column():
                saved_res = gr.Dropdown(
                    label="Resolutions",
                    choices=resolution_manager.get_resolutions(),
                    interactive=True,
                    allow_custom_value=False
                )
            with gr.Column():
                with gr.Row():
                    apply_res_btn = gr.Button(
                        value="Apply resolution", size="lg"
                    )
                    reload_res_btn = gr.Button(
                        value=RELOAD_SYMBOL
                    )
                with gr.Row():
                    del_res_btn = gr.Button(
                        value="Delete resolution", size="lg",
                        variant="stop"
                    )
        with gr.Group():
            with gr.Column():
                new_res = gr.Textbox(
                    label="New Resolution name",
                    placeholder="Resolution preset name"
                )
            with gr.Column():
                save_res_btn = gr.Button(
                    value="Save resolution", size="lg"
                )

    apply_res_btn.click(
        apply_resolution,
        inputs=[saved_res],
        outputs=[width, height]
    )

    save_res_btn.click(
        save_and_refresh_resolution,
        inputs=[new_res, width, height],
        outputs=[saved_res]
    )

    del_res_btn.click(
        delete_and_refresh_resolution,
        inputs=[saved_res],
        outputs=[saved_res]
    )

    reload_res_btn.click(
        refresh_resolution_list,
        inputs=[],
        outputs=[saved_res]
    )

    return {}

"""sd.cpp-webui - UI component for the preset saving feature"""

import gradio as gr

from modules.shared_instance import preset_manager
from .constants import RELOAD_SYMBOL


def create_presets_ui():
    """Create the presets UI layout and return the components"""
    with gr.Row():
        with gr.Accordion(label="Saved presets", open=False):
            with gr.Group():
                with gr.Column():
                    saved_presets = gr.Dropdown(
                        label="Presets",
                        choices=preset_manager.get_presets(),
                        interactive=True,
                        allow_custom_value=False
                    )
                with gr.Column():
                    with gr.Row():
                        load_preset_btn = gr.Button(
                            value="Load preset", size="lg"
                        )
                        reload_presets_btn = gr.Button(
                            value=RELOAD_SYMBOL
                        )
                    with gr.Row():
                        del_preset_btn = gr.Button(
                            value="Delete preset",
                            size="lg",
                            variant="stop"
                        )
            with gr.Group():
                with gr.Column():
                    new_preset = gr.Textbox(
                        label="New Preset name",
                        placeholder="Preset name"
                    )
                with gr.Column():
                    save_preset_btn = gr.Button(
                        value="Save preset", size="lg"
                    )

    return {
        'saved_presets': saved_presets,
        'load_preset_btn': load_preset_btn,
        'reload_presets_btn': reload_presets_btn,
        'del_preset_btn': del_preset_btn,
        'new_preset': new_preset,
        'save_preset_btn': save_preset_btn,
    }


def bind_presets_events(presets_ui, *ui_dicts, preset_flag=None):
    """Keep all click events encapsulated in this file"""

    combined_ui = {}
    for ui_dict in ui_dicts:
        combined_ui.update(ui_dict)

    preset_keys = list(combined_ui.keys())
    preset_components = list(combined_ui.values())

    def save_and_refresh_presets(name, *values):
        if getattr(preset_manager, "is_default", lambda x: False)(name):
            gr.Warning(f"Cannot overwrite the default preset: '{name}'. Please use a different name.")
            return gr.skip()

        settings_dict = dict(zip(preset_keys, values))
        preset_manager.add_preset(name, **settings_dict)
        gr.Info(f"Preset '{name}' saved.")
        return gr.update(choices=preset_manager.get_presets(), value=name)

    def load_selected_preset(preset_name):
        preset = preset_manager.get_preset(preset_name)
        if not preset:
            return [gr.skip()] * len(preset_keys)

        output_values = []
        for key in preset_keys:
            old_key_format = key.replace('in_', '')
            if key in preset:
                val = preset[key]
            elif old_key_format in preset:
                val = preset[old_key_format]
            else:
                val = gr.skip()
            output_values.append(val)

        gr.Info(f"Preset '{preset_name}' loaded.")

        return tuple(output_values)

    def delete_and_refresh_presets(name):
        if getattr(preset_manager, "is_default", lambda x: False)(name):
            gr.Warning(f"Cannot delete default preset: '{name}'.")
            return gr.skip()
        preset_manager.delete_preset(name)
        gr.Info(f"Preset '{name}' deleted.")
        return gr.update(choices=preset_manager.get_presets())

    def refresh_preset_list():
        return gr.update(choices=preset_manager.get_presets())

    presets_ui['save_preset_btn'].click(
        save_and_refresh_presets,
        inputs=[presets_ui['new_preset']] + preset_components,
        outputs=[presets_ui['saved_presets']]
    )

    if preset_flag is not None:
        presets_ui['load_preset_btn'].click(
            fn=lambda: True, inputs=[], outputs=[preset_flag]
        ).then(
            fn=load_selected_preset,
            inputs=[presets_ui['saved_presets']],
            outputs=preset_components
        )
    else:
        presets_ui['load_preset_btn'].click(
            load_selected_preset,
            inputs=[presets_ui['saved_presets']],
            outputs=preset_components
        )

    presets_ui['del_preset_btn'].click(
        delete_and_refresh_presets,
        inputs=[presets_ui['saved_presets']],
        outputs=[presets_ui['saved_presets']]
    )

    presets_ui['reload_presets_btn'].click(
        refresh_preset_list, inputs=[], outputs=[presets_ui['saved_presets']]
    )

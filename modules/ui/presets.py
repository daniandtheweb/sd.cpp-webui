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


def bind_presets_events(presets_ui, generation_settings_ui):
    """Keep all click events encapsulated in this file"""

    preset_map = {
        'sampling': generation_settings_ui['in_sampling'],
        'scheduler': generation_settings_ui['in_scheduler'],
        'width': generation_settings_ui['in_width'],
        'height': generation_settings_ui['in_height'],
        'steps': generation_settings_ui['in_steps'],
        'cfg': generation_settings_ui['in_cfg']
    }

    preset_keys = list(preset_map.keys())
    preset_components = list(preset_map.values())

    def save_and_refresh_presets(name, *values):
        settings_dict = dict(zip(preset_keys, values))
        preset_manager.add_preset(name, **settings_dict)
        return gr.update(choices=preset_manager.get_presets(), value=name)

    def load_selected_preset(preset_name):
        preset = preset_manager.get_preset(preset_name)
        if not preset:
            return [gr.skip()] * len(preset_keys)

        return tuple(preset.get(key, gr.skip()) for key in preset_keys)

    def delete_and_refresh_presets(name):
        preset_manager.delete_preset(name)
        return gr.update(choices=preset_manager.get_presets())

    def refresh_preset_list():
        return gr.update(choices=preset_manager.get_presets())

    presets_ui['save_preset_btn'].click(
        save_and_refresh_presets,
        inputs=[presets_ui['new_preset']] + preset_components,
        outputs=[presets_ui['saved_presets']]
    )

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
        refresh_preset_list,
        inputs=[],
        outputs=[presets_ui['saved_presets']]
    )

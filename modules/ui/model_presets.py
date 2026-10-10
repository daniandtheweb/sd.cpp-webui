"""sd.cpp-webui - UI component for saving and loading model and generation presets"""

import gradio as gr

from modules.shared_instance import preset_manager
from .constants import RELOAD_SYMBOL


def create_model_presets_ui():
    """Create the model presets accordion layout and return the components"""
    with gr.Accordion(label="Presets", open=False):
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


def bind_model_presets_events(
    model_presets_ui, model_inputs, *ui_dicts,
    model_tabs=None, preset_flag=None
):
    """Keep all preset click events encapsulated in this file"""

    combined_ui = dict(model_inputs)
    for ui_dict in ui_dicts:
        combined_ui.update(ui_dict)

    preset_keys = list(combined_ui.keys())
    preset_components = list(combined_ui.values())

    # Classify the model-selection keys so we can drop the irrelevant half
    # of the model set when saving (checkpoint keys vs unet keys).
    # Generation/other settings (from ui_dicts) are always saved.
    model_input_keys = set(model_inputs.keys())
    ckpt_key_set = {k for k in model_input_keys if 'ckpt' in k.lower()}
    unet_key_set = model_input_keys - ckpt_key_set - {'in_diffusion_mode'}

    load_outputs = list(preset_components)
    if model_tabs is not None:
        load_outputs.append(model_tabs)
    if preset_flag is not None:
        load_outputs.append(preset_flag)

    def save_and_refresh_presets(name, *values):
        name = (name or "").strip()
        if not name:
            gr.Warning("Please enter a preset name before saving.")
            return gr.skip()

        if preset_manager.is_default(name):
            gr.Warning(f"Cannot overwrite the default preset: '{name}'. Please use a different name.")
            return gr.skip()

        settings_dict = dict(zip(preset_keys, values))

        # Only store the model parameters for the active tab. When saving in
        # checkpoint mode we drop the unet params, and vice versa, so a preset
        # never carries stale values for the tab it isn't using.
        if 'in_diffusion_mode' in settings_dict:
            mode = settings_dict['in_diffusion_mode']
            if mode == 1:
                for key in ckpt_key_set:
                    settings_dict.pop(key, None)
            elif mode == 0:
                for key in unet_key_set:
                    settings_dict.pop(key, None)

        preset_manager.add_preset(name, **settings_dict)
        gr.Info(f"Preset '{name}' saved.")
        return gr.update(choices=preset_manager.get_presets(), value=name)

    def load_selected_preset(preset_name):
        preset = preset_manager.get_preset(preset_name)
        if not preset:
            result = [gr.skip()] * len(load_outputs)
            if preset_flag is not None:
                result[-1] = False
            return result

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

        if model_tabs is not None:
            mode = preset.get('in_diffusion_mode')
            if mode is not None:
                output_values.append(
                    gr.update(selected="unet" if mode == 1 else "checkpoint")
                )
            else:
                output_values.append(gr.skip())

        # Reset the loading flag here. A programmatic tab switch
        # (gr.update(selected=...)) does NOT fire the tab's .select event,
        # so we cannot rely on the tab-switch handler to clear the flag.
        if preset_flag is not None:
            output_values.append(False)

        gr.Info(f"Preset '{preset_name}' loaded.")

        return tuple(output_values)

    def delete_and_refresh_presets(name):
        if preset_manager.is_default(name):
            gr.Warning(f"Cannot delete default preset: '{name}'.")
            return gr.skip()
        preset_manager.delete_preset(name)
        gr.Info(f"Preset '{name}' deleted.")
        return gr.update(choices=preset_manager.get_presets())

    def refresh_preset_list():
        return gr.update(choices=preset_manager.get_presets())

    model_presets_ui['save_preset_btn'].click(
        save_and_refresh_presets,
        inputs=[model_presets_ui['new_preset']] + preset_components,
        outputs=[model_presets_ui['saved_presets']]
    )

    if preset_flag is not None:
        model_presets_ui['load_preset_btn'].click(
            fn=lambda: True, inputs=[], outputs=[preset_flag]
        ).then(
            fn=load_selected_preset,
            inputs=[model_presets_ui['saved_presets']],
            outputs=load_outputs
        )
    else:
        model_presets_ui['load_preset_btn'].click(
            load_selected_preset,
            inputs=[model_presets_ui['saved_presets']],
            outputs=load_outputs
        )

    model_presets_ui['del_preset_btn'].click(
        delete_and_refresh_presets,
        inputs=[model_presets_ui['saved_presets']],
        outputs=[model_presets_ui['saved_presets']]
    )

    model_presets_ui['reload_presets_btn'].click(
        refresh_preset_list, inputs=[], outputs=[model_presets_ui['saved_presets']]
    )

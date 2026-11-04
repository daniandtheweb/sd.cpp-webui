"""sd.cpp-webui - Settings UI"""

import gradio as gr

from modules.utils.ui_events import refresh_all_options
from modules.shared_instance import (
    config
)
from modules.ui.constants import (
    SORT_OPTIONS, THEMES
)
from modules.ui.models import create_model_widget
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.folder_settings import create_folders_opt_ui
from modules.ui.performance import create_performance_ui
from modules.ui.deprecated import create_deprecated_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.preview import create_preview_ui
from modules.ui.environment import create_env_ui
from modules.options import registry as option_registry


OUTPUT_SCHEMES = ["Sequential", "Timestamp", "TimestampMS", "EpochTime"]

# 'in_<name>' -> 'def_<name>': the config key each setting's default is
# saved under (from the option registry)
CONFIG_KEY_MAP = option_registry.config_key_map()


def config_key_for(ui_key: str) -> str:
    return CONFIG_KEY_MAP.get(ui_key, ui_key)


class SettingsBinder:
    """
    Binds the settings widgets to their config keys: keeps their order and
    saves or resets them.
    """

    def __init__(self, config_obj):
        self.config = config_obj
        self.map = {}
        self.ordered_keys = []
        self.ordered_components = []

    def register(self, key: str, component: gr.components.Component):
        self.map[key] = component
        self.ordered_keys.append(key)
        self.ordered_components.append(component)
        return component

    def save(self, *args) -> str:
        params = dict(zip(self.ordered_keys, args))
        self.config.update_settings(params)
        return "✅ Settings saved successfully."

    def reset(self) -> tuple:
        self.config.reset_defaults()
        status = "🔄 Defaults restored."
        # Refresh every registered widget with the restored defaults,
        # otherwise stale values are re-saved on the next "Save defaults".
        updates = [
            gr.update(value=self.config.get(key))
            for key in self.ordered_keys
        ]
        return (status, *updates)


# Binds the widgets to the config
binder = SettingsBinder(config)


with gr.Blocks() as settings_block:
    settings_map = {}
    # Title
    settings_title = gr.Markdown("# Settings")

    with gr.Tab(label="Default models"):
        with gr.Row():
            binder.register(
                'def_model_tab', gr.Radio(
                    label="Default model TAB",
                    choices=["checkpoint", "unet"],
                    value=config.get('def_model_tab'))
                )
        with gr.Row():
            binder.register(
                'def_ckpt', create_model_widget(
                    label="Checkpoint Model",
                    dir_key='ckpt_dir',
                    config_key='def_ckpt')
                )
            binder.register(
                'def_ckpt_vae', create_model_widget(
                    label="Checkpoint VAE",
                    dir_key='vae_dir',
                    config_key='def_ckpt_vae')
                )
        with gr.Row():
            binder.register(
                'def_unet', create_model_widget(
                    label="UNET Model",
                    dir_key='unet_dir',
                    config_key='def_unet')
                )
            binder.register(
                'def_unet_vae', create_model_widget(
                    label="UNET VAE",
                    dir_key='vae_dir',
                    config_key='def_unet_vae')
                )
        with gr.Row():
            binder.register(
                'def_clip_g', create_model_widget(
                    label="clip_g",
                    dir_key='txt_enc_dir',
                    config_key='def_clip_g')
                )
            binder.register(
                'def_clip_l', create_model_widget(
                    label="clip_l",
                    dir_key='txt_enc_dir',
                    config_key='def_clip_l')
                )
            binder.register(
                'def_clip_vision_h', create_model_widget(
                    label="clip_vision_h",
                    dir_key='txt_enc_dir',
                    config_key='def_clip_vision_h')
                )
        with gr.Row():
            binder.register(
                'def_t5xxl', create_model_widget(
                    label="t5xxl",
                    dir_key='txt_enc_dir',
                    config_key='def_t5xxl')
                )
            binder.register(
                'def_umt5_xxl', create_model_widget(
                    label="umt5_xxl",
                    dir_key='txt_enc_dir',
                    config_key='def_umt5_xxl')
                )
        with gr.Row():
            binder.register(
                'def_llm', create_model_widget(
                    label="llm",
                    dir_key='txt_enc_dir',
                    config_key='def_llm')
                )
            binder.register(
                'def_llm_vision', create_model_widget(
                    label="llm_vision",
                    dir_key='txt_enc_dir',
                    config_key='def_llm_vision')
                )
        with gr.Row():
            binder.register(
                'def_tokenizer', create_model_widget(
                    label="tokenizer",
                    dir_key='tokenizer_dir',
                    config_key='def_tokenizer')
                )

        quant_ui = create_quant_ui()
        for k, v in quant_ui.items():
            binder.register(config_key_for(k), v)

    with gr.Tab(label="Generation settings"):

        generation_settings_ui = create_generation_settings_ui(True)
        binder.register(
            'def_sampling', generation_settings_ui['in_sampling']
        )
        binder.register(
            'def_steps', generation_settings_ui['in_steps']
        )
        binder.register(
            'def_scheduler', generation_settings_ui['in_scheduler']
        )
        binder.register(
            'def_width', generation_settings_ui['in_width']
        )
        binder.register(
            'def_height', generation_settings_ui['in_height']
        )
        binder.register(
            'def_cfg', generation_settings_ui['in_cfg']
        )
        binder.register(
            'def_guidance_bool', generation_settings_ui['in_guidance_bool']
        )
        binder.register(
            'def_guidance', generation_settings_ui['in_guidance']
        )
        binder.register(
            'def_flow_shift_bool', generation_settings_ui['in_flow_shift_bool']
        )
        binder.register(
            'def_flow_shift', generation_settings_ui['in_flow_shift']
        )

        bottom_generation_settings_ui = create_bottom_generation_settings_ui()
        binder.register(
            'def_seed', bottom_generation_settings_ui['in_seed']
        )
        binder.register(
            'def_clip_skip', bottom_generation_settings_ui['in_clip_skip']
        )
        binder.register(
            'def_batch_count', bottom_generation_settings_ui['in_batch_count']
        )

    with gr.Tab(label="Advanced settings"):

        for prefix, ui_dict in [
            ('taesd', create_taesd_ui()),
            ('vae_tiling', create_vae_tiling_ui()),
            ('cache', create_cache_ui()),
            ('extras', create_extras_ui()),
            ('preview', create_preview_ui()),
            ('performance', create_performance_ui()),
            ('env', create_env_ui()),
            ('deprecated', create_deprecated_ui())
        ]:
            for k, v in ui_dict.items():
                binder.register(config_key_for(k), v)

    with gr.Tab(label="Directories"):

        # Folders settings
        folders_ui = create_folders_opt_ui()
        for k, v in folders_ui.items():
            binder.register(k.replace('_txt', ''), v)

    with gr.Tab(label="sd.cpp-webui settings"):

        with gr.Row():
            # Output settings
            binder.register(
                'def_output_scheme', gr.Dropdown(
                    label="Output Scheme",
                    choices=OUTPUT_SCHEMES,
                    value=config.get('def_output_scheme'),
                    interactive=True
                )
            )

            binder.register(
                'def_output_steps', gr.Checkbox(
                    label="Add steps count to the output name",
                    value=config.get('def_output_steps'),
                    interactive=True
                )
            )

            binder.register(
                'def_output_quant', gr.Checkbox(
                    label="Add quantization type to the output name",
                    value=config.get('def_output_quant'),
                    interactive=True
                )
            )

        with gr.Row():
            # Gallery settings
            binder.register(
                'def_gallery_sorting', gr.Radio(
                    label="Sort By",
                    choices=SORT_OPTIONS,
                    value=config.get('def_gallery_sorting'),
                    interactive=True
                )
            )

        with gr.Row():
            # Theme settings
            binder.register(
                'def_theme', gr.Dropdown(
                    label="Theme",
                    choices=THEMES,
                    value=config.get('def_theme'),
                    interactive=True
                )
            )
            binder.register(
                'def_color_scheme', gr.Dropdown(
                    label="Color scheme",
                    choices=["System", "Light", "Dark"],
                    value=(config.get('def_color_scheme') or 'System').title(),
                    interactive=True
                )
            )

    with gr.Row():
        refresh_choices = gr.Button(
            value="Refresh sd choices"
        )

    with gr.Group():
        status_textbox = gr.Textbox(
            label="Status",
            value="",
            interactive=False
        )

        # Save defaults and Restore defaults buttons
        with gr.Row():
            save_btn = gr.Button(
                value="Save defaults", variant="primary"
            )
            restore_btn = gr.Button(
                value="Restore defaults", variant="stop"
            )
        with gr.Row():
            restart_btn = gr.Button(
                value="Restart server", variant="stop"
            )

    ordered_keys = binder.ordered_keys
    ordered_components = binder.ordered_components

    save_btn.click(
        binder.save,
        inputs=ordered_components,
        outputs=[status_textbox]
    )

    restore_btn.click(
        binder.reset,
        inputs=[],
        outputs=[status_textbox] + ordered_components
    )

    # Safely collect refresh targets
    refresh_outputs = [
        binder.map.get('def_sampling'),
        binder.map.get('def_scheduler'),
        binder.map.get('def_preview_mode'),
        binder.map.get('def_predict'),
    ]

    # Remove any missing components to prevent Gradio initialization crash
    missing_refresh_outputs = [
        name for name, component in zip(
            ['def_sampling', 'def_scheduler', 'def_preview_mode', 'def_predict'],
            refresh_outputs
        )
        if component is None
    ]

    if missing_refresh_outputs:
        raise RuntimeError(
            f"Missing refresh output component(s): {missing_refresh_outputs}"
        )

    refresh_choices.click(
        refresh_all_options,
        inputs=[],
        outputs=refresh_outputs
    )

"""sd.cpp-webui - Options UI"""

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
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.preview import create_preview_ui
from modules.ui.environment import create_env_ui


OUTPUT_SCHEMES = ["Sequential", "Timestamp", "TimestampMS", "EpochTime"]

OPTION_KEY_MAP = {
    # Quant
    'in_model_type': 'def_model_type',
    'in_tensor_type_rules': 'def_tensor_type_rules',

    # TAESD
    'in_taesd': 'def_taesd',

    # VAE tiling
    'in_vae_tiling': 'def_vae_tiling',
    'in_vae_tile_overlap': 'def_vae_tile_overlap',
    'in_vae_tile_size': 'def_vae_tile_size',
    'in_vae_relative_bool': 'def_vae_relative_bool',
    'in_vae_relative_tile_size': 'def_vae_relative_tile_size',
    'in_temporal_tiling': 'def_temporal_tiling',

    # Cache
    'in_cache_bool': 'def_cache_bool',
    'in_cache_mode': 'def_cache_mode',
    'in_cache_option': 'def_cache_option',
    'in_scm_mask': 'def_scm_mask',
    'in_scm_policy': 'def_scm_policy',

    # Extras
    'in_sigmas': 'def_sigmas',
    'in_rng': 'def_rng',
    'in_sampler_rng': 'def_sampler_rng',
    'in_predict': 'def_predict',
    'in_lora_apply': 'def_lora_apply',
    'in_output': 'def_output',
    'in_mmap': 'def_mmap',
    'in_color': 'def_color',
    'in_verbose': 'def_verbose',

    # Preview
    'in_preview_bool': 'def_preview_bool',
    'in_preview_mode': 'def_preview_mode',
    'in_preview_interval': 'def_preview_interval',
    'in_preview_taesd': 'def_preview_taesd',
    'in_preview_noisy': 'def_preview_noisy',

    # Performance
    'in_threads': 'def_threads',
    'in_max_vram': 'def_max_vram',
    'in_offload_to_cpu': 'def_offload_to_cpu',
    'in_vae_cpu': 'def_vae_cpu',
    'in_clip_cpu': 'def_clip_cpu',
    'in_flash_attn': 'def_flash_attn',
    'in_diffusion_fa': 'def_diffusion_fa',
    'in_diffusion_conv_direct': 'def_diffusion_conv_direct',
    'in_vae_conv_direct': 'def_vae_conv_direct',
    'in_force_sdxl_vae_conv_scale': 'def_force_sdxl_vae_conv_scale',

    # Environment
    'env_vk_visible_override': 'def_env_vk_visible_override',
    'env_GGML_VK_VISIBLE_DEVICES': 'def_env_GGML_VK_VISIBLE_DEVICES',
    'env_cuda_visible_override': 'def_env_cuda_visible_override',
    'env_CUDA_VISIBLE_DEVICES': 'def_env_CUDA_VISIBLE_DEVICES',
    'env_GGML_VK_DISABLE_COOPMAT': 'def_env_GGML_VK_DISABLE_COOPMAT',
    'env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': 'def_env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT'}


def resolve_option_key(ui_key: str) -> str:
    return OPTION_KEY_MAP.get(ui_key, ui_key)


class SettingsRegistry:
    """Manages UI-to-config mapping, saving, and ordering."""

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

    def reset(self) -> str:
        self.config.reset_defaults()
        return "🔄 Defaults restored."


# Initialize registry
registry = SettingsRegistry(config)


with gr.Blocks() as options_block:
    settings_map = {}
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Tab(label="Default models"):
        with gr.Row():
            registry.register(
                'def_model_tab', gr.Radio(
                    label="Default model TAB",
                    choices=["checkpoint", "unet"],
                    value=config.get('def_model_tab'))
                )
        with gr.Row():
            registry.register(
                'def_ckpt', create_model_widget(
                    label="Checkpoint Model",
                    dir_key='ckpt_dir',
                    option_key='def_ckpt')
                )
            registry.register(
                'def_ckpt_vae', create_model_widget(
                    label="Checkpoint VAE",
                    dir_key='vae_dir',
                    option_key='def_ckpt_vae')
                )
        with gr.Row():
            registry.register(
                'def_unet', create_model_widget(
                    label="UNET Model",
                    dir_key='unet_dir',
                    option_key='def_unet')
                )
            registry.register(
                'def_unet_vae', create_model_widget(
                    label="UNET VAE",
                    dir_key='vae_dir',
                    option_key='def_unet_vae')
                )
        with gr.Row():
            registry.register(
                'def_clip_g', create_model_widget(
                    label="clip_g",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_g')
                )
            registry.register(
                'def_clip_l', create_model_widget(
                    label="clip_l",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_l')
                )
            registry.register(
                'def_clip_vision_h', create_model_widget(
                    label="clip_vision_h",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_vision_h')
                )
        with gr.Row():
            registry.register(
                'def_t5xxl', create_model_widget(
                    label="t5xxl",
                    dir_key='txt_enc_dir',
                    option_key='def_t5xxl')
                )
            registry.register(
                'def_umt5_xxl', create_model_widget(
                    label="umt5_xxl",
                    dir_key='txt_enc_dir',
                    option_key='def_umt5_xxl')
                )
        with gr.Row():
            registry.register(
                'def_llm', create_model_widget(
                    label="llm",
                    dir_key='txt_enc_dir',
                    option_key='def_llm')
                )

        quant_ui = create_quant_ui()
        for k, v in quant_ui.items():
            registry.register(resolve_option_key(k), v)

    with gr.Tab(label="Generation settings"):

        generation_settings_ui = create_generation_settings_ui(True)
        registry.register(
            'def_sampling', generation_settings_ui['in_sampling']
        )
        registry.register(
            'def_steps', generation_settings_ui['in_steps']
        )
        registry.register(
            'def_scheduler', generation_settings_ui['in_scheduler']
        )
        registry.register(
            'def_width', generation_settings_ui['in_width']
        )
        registry.register(
            'def_height', generation_settings_ui['in_height']
        )
        registry.register(
            'def_cfg', generation_settings_ui['in_cfg']
        )
        registry.register(
            'def_guidance_bool', generation_settings_ui['in_guidance_bool']
        )
        registry.register(
            'def_guidance', generation_settings_ui['in_guidance']
        )
        registry.register(
            'def_flow_shift_bool', generation_settings_ui['in_flow_shift_bool']
        )
        registry.register(
            'def_flow_shift', generation_settings_ui['in_flow_shift']
        )

        bottom_generation_settings_ui = create_bottom_generation_settings_ui()
        registry.register(
            'def_seed', bottom_generation_settings_ui['in_seed']
        )
        registry.register(
            'def_clip_skip', bottom_generation_settings_ui['in_clip_skip']
        )
        registry.register(
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
            ('env', create_env_ui())
        ]:
            for k, v in ui_dict.items():
                registry.register(resolve_option_key(k), v)

    with gr.Tab(label="Directories"):

        # Folders options
        folders_ui = create_folders_opt_ui()
        for k, v in folders_ui.items():
            registry.register(k.replace('_txt', ''), v)

    with gr.Tab(label="sd.cpp-webui settings"):

        with gr.Row():
            # Output options
            registry.register(
                'def_output_scheme', gr.Dropdown(
                    label="Output Scheme",
                    choices=OUTPUT_SCHEMES,
                    value=config.get('def_output_scheme'),
                    interactive=True
                )
            )

            registry.register(
                'def_output_steps', gr.Checkbox(
                    label="Add steps count to the output name",
                    value=config.get('def_output_steps'),
                    interactive=True
                )
            )

            registry.register(
                'def_output_quant', gr.Checkbox(
                    label="Add quantization type to the output name",
                    value=config.get('def_output_quant'),
                    interactive=True
                )
            )

        with gr.Row():
            # Gallery options
            registry.register(
                'def_gallery_sorting', gr.Radio(
                    label="Sort By",
                    choices=SORT_OPTIONS,
                    value=config.get('def_gallery_sorting'),
                    interactive=True
                )
            )

        with gr.Row():
            # Theme options
            registry.register(
                'def_theme', gr.Dropdown(
                    label="Theme:",
                    choices=THEMES,
                    value=config.get('def_theme'),
                    interactive=True
                )
            )

    with gr.Row():
        refresh_opt = gr.Button(
            value="Refresh sd options"
        )

    with gr.Group():
        status_textbox = gr.Textbox(
            label="Status",
            value="",
            interactive=False
        )

        # Set Defaults and Restore Defaults Buttons
        with gr.Row():
            set_btn = gr.Button(
                value="Set new options", variant="primary"
            )
            restore_btn = gr.Button(
                value="Restore Defaults", variant="stop"
            )
        with gr.Row():
            restart_btn = gr.Button(
                value="Restart server", variant="stop"
            )

    ordered_keys = registry.ordered_keys
    ordered_components = registry.ordered_components

    set_btn.click(
        registry.save,
        inputs=ordered_components,
        outputs=[status_textbox]
    )

    restore_btn.click(
        registry.reset,
        inputs=[],
        outputs=[status_textbox]
    )

    # Safely collect refresh targets
    refresh_outputs = [
        registry.map.get('def_sampling'),
        registry.map.get('def_scheduler'),
        registry.map.get('def_preview_mode'),
        registry.map.get('def_predict'),
    ]

    # Remove any missing components to prevent Gradio initialization crash
    missing_refresh_outputs = [
        name for name, component in zip(
            ['def_sampling', 'def_scheduler', 'def_preview_mode', 'predict'],
            refresh_outputs
        )
        if component is None
    ]

    if missing_refresh_outputs:
        raise RuntimeError(
            f"Missing refresh output component(s): {missing_refresh_outputs}"
        )

    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=refresh_outputs
    )

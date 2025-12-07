"""sd.cpp-webui - Options UI"""

import gradio as gr

from modules.shared_instance import (
    config, sd_options
)
from modules.ui.constants import (
    PREDICTION
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
from modules.ui.easycache import create_easycache_ui
from modules.ui.preview import create_preview_ui
from modules.ui.environment import create_env_ui


OUTPUT_SCHEMES = ["Sequential", "Timestamp", "TimestampMS", "EpochTime"]


def refresh_all_options():
    """Updates the available options from the sd executable."""
    sd_options.refresh()
    return [
        gr.update(choices=sd_options.get_opt("samplers")),
        gr.update(choices=sd_options.get_opt("schedulers")),
        gr.update(choices=sd_options.get_opt("prediction"))
    ]


def save_settings_wrapper(*args):
    """Gathers all UI values into a dictionary and calls the config manager."""
    params = dict(zip(ordered_keys, args))

    config.update_settings(params)
    return "Settings saved successfully."


with gr.Blocks() as options_block:
    settings_map = {}
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Accordion(
        label="Default models", open=False
    ):
        with gr.Row():
            with gr.Column():
                ckpt_model = create_model_widget(
                    label="Checkpoint Model",
                    dir_key='ckpt_dir',
                    option_key='def_ckpt',
                )
                settings_map['def_ckpt'] = ckpt_model
            with gr.Column():
                ckpt_vae = create_model_widget(
                    label="Checkpoint VAE",
                    dir_key='vae_dir',
                    option_key='def_ckpt_vae',
                )
                settings_map['def_ckpt_vae'] = ckpt_vae

        with gr.Row():
            with gr.Column():
                unet_model = create_model_widget(
                    label="UNET Model",
                    dir_key='unet_dir',
                    option_key='def_unet',
                )
                settings_map['def_unet'] = unet_model
            with gr.Column():
                unet_vae = create_model_widget(
                    label="UNET VAE",
                    dir_key='vae_dir',
                    option_key='def_unet_vae',
                )
                settings_map['def_unet_vae'] = unet_vae

        with gr.Row():
            with gr.Column():
                clip_g = create_model_widget(
                    label="clip_g",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_g',
                )
                settings_map['def_clip_g'] = clip_g
            with gr.Column():
                clip_l = create_model_widget(
                    label="clip_l",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_l',
                )
                settings_map['def_clip_l'] = clip_l
            with gr.Column():
                clip_vision_h = create_model_widget(
                    label="clip_vision_h",
                    dir_key='txt_enc_dir',
                    option_key='def_clip_vision_h',
                )
                settings_map['def_clip_vision_h'] = clip_vision_h

        with gr.Row():
            with gr.Column():
                t5xxl = create_model_widget(
                    label="t5xxl",
                    dir_key='txt_enc_dir',
                    option_key='def_t5xxl',
                )
                settings_map['def_t5xxl'] = t5xxl
            with gr.Column():
                umt5_xxl = create_model_widget(
                    label="umt5_xxl",
                    dir_key='txt_enc_dir',
                    option_key='def_umt5_xxl',
                )
                settings_map['def_umt5_xxl'] = umt5_xxl

        with gr.Row():
            with gr.Column():
                llm = create_model_widget(
                    label="llm",
                    dir_key='txt_enc_dir',
                    option_key='def_llm',
                )
                settings_map['def_llm'] = llm

    # Model Type Selection
    quant_ui = create_quant_ui()
    settings_map.update(quant_ui)

    generation_settings_ui = create_generation_settings_ui(True)
    settings_map.update({
        'def_sampling': generation_settings_ui['in_sampling'],
        'def_steps': generation_settings_ui['in_steps'],
        'def_scheduler': generation_settings_ui['in_scheduler'],
        'def_width': generation_settings_ui['in_width'],
        'def_height': generation_settings_ui['in_height'],
        'def_cfg': generation_settings_ui['in_cfg'],
        'def_guidance_bool': generation_settings_ui['in_guidance_bool'],
        'def_guidance': generation_settings_ui['in_guidance'],
        'def_flow_shift_bool': generation_settings_ui['in_flow_shift_bool'],
        'def_flow_shift': generation_settings_ui['in_flow_shift']
    })

    bottom_generation_settings_ui = create_bottom_generation_settings_ui()
    settings_map.update({
        'def_clip_skip': bottom_generation_settings_ui['in_clip_skip'],
        'def_batch_count': bottom_generation_settings_ui['in_batch_count']
    })

    with gr.Row():
        # Prediction mode
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict'),
            interactive=True
        )
        settings_map['predict'] = predict

    taesd_ui = create_taesd_ui()
    settings_map.update({
        'def_taesd': taesd_ui['in_taesd']
    })

    vae_tiling_ui = create_vae_tiling_ui()
    settings_map.update({
        'def_vae_tiling': vae_tiling_ui['in_vae_tiling'],
        'def_vae_tile_overlap': vae_tiling_ui['in_vae_tile_overlap'],
        'def_vae_tile_size': vae_tiling_ui['in_vae_tile_size'],
        'def_vae_relative_bool': vae_tiling_ui['in_vae_relative_bool'],
        'def_vae_relative_tile_size': vae_tiling_ui['in_vae_relative_tile_size']
    })

    easycache_ui = create_easycache_ui()
    settings_map.update({
        'def_easycache_bool': easycache_ui['in_easycache_bool'],
        'def_ec_threshold': easycache_ui['in_ec_threshold'],
        'def_ec_start': easycache_ui['in_ec_start'],
        'def_ec_end': easycache_ui['in_ec_end']
    })

    preview_ui = create_preview_ui()
    settings_map.update({
        'def_preview_bool': preview_ui['in_preview_bool'],
        'def_preview_mode': preview_ui['in_preview_mode'],
        'def_preview_interval': preview_ui['in_preview_interval'],
        'def_preview_taesd': preview_ui['in_preview_taesd'],
        'def_preview_noisy': preview_ui['in_preview_noisy']
    })

    performance_ui = create_performance_ui()
    settings_map.update({
        'def_threads': performance_ui['in_threads'],
        'def_offload_to_cpu': performance_ui['in_offload_to_cpu'],
        'def_vae_cpu': performance_ui['in_vae_cpu'],
        'def_clip_cpu': performance_ui['in_clip_cpu'],
        'def_flash_attn': performance_ui['in_flash_attn'],
        'def_diffusion_conv_direct': performance_ui['in_diffusion_conv_direct'],
        'def_vae_conv_direct': performance_ui['in_vae_conv_direct'],
        'def_force_sdxl_vae_conv_scale': performance_ui['in_force_sdxl_vae_conv_scale']
    })

    env_ui = create_env_ui()
    settings_map.update({
        'def_env_vk_visible_override': env_ui['env_vk_visible_override'],
        'def_env_GGML_VK_VISIBLE_DEVICES': env_ui['env_GGML_VK_VISIBLE_DEVICES'],
        'def_env_cuda_visible_override': env_ui['env_cuda_visible_override'],
        'def_env_CUDA_VISIBLE_DEVICES': env_ui['env_CUDA_VISIBLE_DEVICES'],
        'def_env_GGML_VK_DISABLE_COOPMAT': env_ui['env_GGML_VK_DISABLE_COOPMAT'],
        'def_env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT': env_ui['env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT']
    })

    with gr.Row():
        # Output options
        output_scheme = gr.Dropdown(
            label="Output Scheme",
            choices=OUTPUT_SCHEMES,
            value=config.get('def_output_scheme'),
            interactive=True
        )
        settings_map['def_output_scheme'] = output_scheme

    # Folders options
    folders_ui = create_folders_opt_ui()
    settings_map.update({
        'ckpt_dir': folders_ui['ckpt_dir_txt'],
        'unet_dir': folders_ui['unet_dir_txt'],
        'vae_dir': folders_ui['vae_dir_txt'],
        'txt_enc_dir': folders_ui['txt_enc_dir_txt'],
        'emb_dir': folders_ui['emb_dir_txt'],
        'lora_dir': folders_ui['lora_dir_txt'],
        'taesd_dir': folders_ui['taesd_dir_txt'],
        'phtmkr_dir': folders_ui['phtmkr_dir_txt'],
        'upscl_dir': folders_ui['upscl_dir_txt'],
        'cnnet_dir': folders_ui['cnnet_dir_txt'],
        'txt2img_dir': folders_ui['txt2img_dir_txt'],
        'img2img_dir': folders_ui['img2img_dir_txt'],
        'any2video_dir': folders_ui['any2video_dir_txt']
    })

    with gr.Row():
        refresh_opt = gr.Button(
            value="Refresh sd options"
        )

    with gr.Group():
        status_textbox = gr.Textbox(label="Status", value="", interactive=False)

        # Set Defaults and Restore Defaults Buttons
        with gr.Row():
            set_btn = gr.Button(
                value="Set new options", variant="primary"
            )
            restore_btn = gr.Button(
                value="Restore Defaults", variant="stop"
            )

    ordered_keys = sorted(settings_map.keys())
    ordered_components = [settings_map[key] for key in ordered_keys]

    set_btn.click(
        save_settings_wrapper,
        inputs=ordered_components,
        outputs=[status_textbox]
    )

    restore_btn.click(
        config.reset_defaults,
        inputs=[],
        outputs=[status_textbox]
    )

    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            predict
        ]
    )

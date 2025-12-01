"""sd.cpp-webui - Options UI"""

import gradio as gr

from modules.shared_instance import (
    config, sd_options
)
from modules.loader import get_models
from modules.ui.constants import (
    RELOAD_SYMBOL, SAMPLERS, SCHEDULERS, PREDICTION, PREVIEW
)
from modules.ui.generation_settings import create_quant_ui
from modules.ui.folder_settings import create_folders_opt_ui


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

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    ckpt_model = gr.Dropdown(
                        label="Checkpoint Model",
                        choices=get_models(config.get('ckpt_dir')),
                        scale=7,
                        value=config.get('def_ckpt'),
                        interactive=True
                    )
                with gr.Row():
                    reload_ckpt_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        ckpt_model, scale=1
                    )
                settings_map['def_ckpt'] = ckpt_model

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    ckpt_vae = gr.Dropdown(
                        label="Checkpoint VAE",
                        choices=get_models(config.get('vae_dir')),
                        scale=7, value=config.get('def_ckpt_vae'),
                        interactive=True
                    )
                with gr.Row():
                    reload_vae_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        ckpt_vae, scale=1
                    )
                settings_map['def_ckpt_vae'] = ckpt_vae

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    unet_model = gr.Dropdown(
                        label="UNET Model",
                        choices=get_models(config.get('unet_dir')),
                        scale=7,
                        value=config.get('def_unet'),
                        interactive=True
                    )
                with gr.Row():
                    reload_unet_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        unet_model, scale=1
                    )
                settings_map['def_unet'] = unet_model

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    unet_vae = gr.Dropdown(
                        label="UNET VAE",
                        choices=get_models(config.get('vae_dir')),
                        scale=7, value=config.get('def_unet_vae'),
                        interactive=True
                    )
                with gr.Row():
                    reload_vae_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        unet_vae, scale=1
                    )
                settings_map['def_unet_vae'] = unet_vae

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    clip_g = gr.Dropdown(
                        label="clip_g",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_clip_g'),
                        interactive=True
                    )
                with gr.Row():
                    reload_clip_g_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        clip_g, scale=1
                    )
                settings_map['def_clip_g'] = clip_g

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    clip_l = gr.Dropdown(
                        label="clip_l",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_clip_l'),
                        interactive=True
                    )
                with gr.Row():
                    reload_clip_l_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        clip_l, scale=1
                    )
                settings_map['def_clip_l'] = clip_l

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    clip_vision_h = gr.Dropdown(
                        label="clip_vision_h",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_clip_vision_h'),
                        interactive=True
                    )
                with gr.Row():
                    reload_clip_vision_h_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        clip_vision_h, scale=1
                    )
                settings_map['def_clip_vision_h'] = clip_vision_h

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    t5xxl = gr.Dropdown(
                        label="t5xxl",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_t5xxl'),
                        interactive=True
                    )
                with gr.Row():
                    reload_t5xxl_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        t5xxl, scale=1
                    )
                settings_map['def_t5xxl'] = t5xxl

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    umt5_xxl = gr.Dropdown(
                        label="umt5_xxl",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_umt5_xxl'),
                        interactive=True
                    )
                with gr.Row():
                    reload_umt5_xxl_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        umt5_xxl, scale=1
                    )
                settings_map['def_umt5_xxl'] = umt5_xxl

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    llm = gr.Dropdown(
                        label="llm",
                        choices=get_models(config.get('clip_dir')),
                        scale=7,
                        value=config.get('def_llm'),
                        interactive=True
                    )
                with gr.Row():
                    reload_llm_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        llm, scale=1
                    )
                settings_map['def_llm'] = llm

    # Model Type Selection
    quant_ui = create_quant_ui()
    settings_map.update(quant_ui)

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    taesd = gr.Dropdown(
                        label="TAESD",
                        choices=get_models(config.get('taesd_dir')),
                        value=config.get('def_taesd'),
                        interactive=True
                    )
                with gr.Row():
                    reload_taesd_btn = gr.Button(
                        value=RELOAD_SYMBOL, scale=1
                    )
                    gr.ClearButton(
                        taesd, scale=1
                    )
                settings_map['def_taesd'] = taesd


    with gr.Row():
        with gr.Column():
            # Sampling Method Dropdown
            sampling = gr.Dropdown(
                label="Sampling method",
                choices=SAMPLERS,
                value=config.get('def_sampling'),
                interactive=True
            )
            settings_map['def_sampling'] = sampling

        with gr.Column():
            # Steps Slider
            steps = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=99,
                value=config.get('def_steps'),
                step=1
            )
            settings_map['def_steps'] = steps

    with gr.Row():
        with gr.Column():
            # Scheduler Dropdown
            scheduler = gr.Dropdown(
                label="Scheduler",
                choices=SCHEDULERS,
                value=config.get('def_scheduler'),
                interactive=True
            )
            settings_map['def_scheduler'] = scheduler

        with gr.Column():
            # CFG Slider
            cfg = gr.Slider(
                label="CFG Scale",
                minimum=1,
                maximum=30,
                value=config.get('def_cfg'),
                step=0.1,
                interactive=True
            )
            settings_map['def_cfg'] = cfg

    with gr.Column():
        # Size Sliders
        width = gr.Slider(
            label="Width",
            minimum=64,
            maximum=2048,
            value=config.get('def_width'),
            step=8
        )
        settings_map['def_height'] = width

        height = gr.Slider(
            label="Height",
            minimum=64,
            maximum=2048,
            value=config.get('def_height'),
            step=8
        )
        settings_map['def_width'] = height

    with gr.Row():
        # Prediction mode
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=config.get('def_predict'),
            interactive=True
        )
        settings_map['predict'] = predict

    with gr.Row():
        # Boolean options
        flash_attn = gr.Checkbox(
            label="Flash Attention",
            value=config.get('def_flash_attn')
        )
        settings_map['def_flash_attn'] = flash_attn

        diffusion_conv_direct = gr.Checkbox(
            label="Conv2D Direct for diffusion",
            value=config.get('def_diffusion_conv_direct')
        )
        settings_map['def_diffusion_conv_direct'] = diffusion_conv_direct

        vae_conv_direct = gr.Checkbox(
            label="Conv2D Direct for VAE",
            value=config.get('def_vae_conv_direct')
        )
        settings_map['def_vae_conv_direct'] = vae_conv_direct

    with gr.Row():
        # Preview options
        preview_bool = gr.Checkbox(
            label="Enable Preview",
            value=config.get('def_preview_bool')
        )
        settings_map['def_preview_bool'] = preview_bool

        preview_mode = gr.Dropdown(
            label="Preview mode",
            choices=PREVIEW,
            value=config.get('def_preview_mode')
        )
        settings_map['def_preview_mode'] = preview_mode

        preview_interval = gr.Number(
            label="Preview interval",
            value=config.get('def_preview_interval'),
            minimum=1,
            interactive=True
        )
        settings_map['def_preview_interval'] = preview_interval

        preview_taesd = gr.Checkbox(
            label="TAESD for preview only",
            value=config.get('def_preview_taesd')
        )
        settings_map['def_preview_taesd'] = preview_taesd

        preview_noisy = gr.Checkbox(
            label="Preview noisy",
            value=config.get('def_preview_noisy')
        )
        settings_map['def_preview_noisy'] = preview_noisy

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
    settings_map.update(folders_ui)

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
        outputs=[sampling, scheduler, predict]
    )

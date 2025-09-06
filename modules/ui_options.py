"""sd.cpp-webui - Options UI"""

import gradio as gr

from modules.config import (
    set_defaults, rst_def, ckpt_dir, vae_dir, unet_dir, clip_dir,
    def_ckpt, def_ckpt_vae, def_unet, def_unet_vae, def_clip_g,
    def_clip_l, def_t5xxl, def_type, def_sampling, def_steps,
    def_scheduler, def_width, def_height, def_predict,
    def_flash_attn, def_diffusion_conv_direct, def_vae_conv_direct
)
from modules.loader import (
    get_models
)
from modules.ui import (
    create_folders_opt_ui, RELOAD_SYMBOL, QUANTS, SAMPLERS,
    SCHEDULERS, PREDICTION
)


with gr.Blocks() as options_block:
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                ckpt_model = gr.Dropdown(
                    label="Checkpoint Model",
                    choices=get_models(ckpt_dir),
                    scale=7,
                    value=def_ckpt,
                    interactive=True
                )
            with gr.Row():
                reload_ckpt_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_ckpt_model = gr.ClearButton(
                    ckpt_model, scale=1
                )
        with gr.Column():
            with gr.Row():
                ckpt_vae = gr.Dropdown(
                    label="Checkpoint VAE",
                    choices=get_models(vae_dir),
                    scale=7, value=def_ckpt_vae,
                    interactive=True
                )
            with gr.Row():
                reload_vae_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_vae = gr.ClearButton(
                    ckpt_vae, scale=1
                )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                unet_model = gr.Dropdown(
                    label="UNET Model",
                    choices=get_models(unet_dir),
                    scale=7,
                    value=def_unet,
                    interactive=True
                )
            with gr.Row():
                reload_unet_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_unet_model = gr.ClearButton(
                    unet_model, scale=1
                )
        with gr.Column():
            with gr.Row():
                unet_vae = gr.Dropdown(
                    label="UNET VAE",
                    choices=get_models(vae_dir),
                    scale=7, value=def_unet_vae,
                    interactive=True
                )
            with gr.Row():
                reload_vae_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_unet_vae = gr.ClearButton(
                    unet_vae, scale=1
                )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                clip_g = gr.Dropdown(
                    label="clip_g",
                    choices=get_models(clip_dir),
                    scale=7,
                    value=def_clip_g,
                    interactive=True
                )
            with gr.Row():
                reload_clip_g_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_clip_g = gr.ClearButton(
                    clip_g, scale=1    
                )
        with gr.Column():
            with gr.Row():
                clip_l = gr.Dropdown(
                    label="clip_l",
                    choices=get_models(clip_dir),
                    scale=7,
                    value=def_clip_l,
                    interactive=True
                )
            with gr.Row():
                reload_clip_l_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_clip_l = gr.ClearButton(
                    clip_l, scale=1
                )
        with gr.Column():
            with gr.Row():
                t5xxl = gr.Dropdown(
                    label="t5xxl",
                    choices=get_models(clip_dir),
                    scale=7,
                    value=def_t5xxl,
                    interactive=True
                )
            with gr.Row():
                reload_t5xxl_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_t5xxl = gr.ClearButton(
                    t5xxl, scale=1
                )
    model_type = gr.Dropdown(
            label="Quantization",
            choices=QUANTS,
            value=def_type,
            interactive=True
    )
    with gr.Row():
        with gr.Column():
            # Sampling Method Dropdown
            sampling = gr.Dropdown(
                label="Sampling method",
                choices=SAMPLERS,
                value=def_sampling,
                interactive=True
            )
        with gr.Column():
            # Steps Slider
            steps = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=99,
                value=def_steps,
                step=1
            )

    with gr.Row():
        # Scheduler Dropdown
        scheduler = gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULERS,
            value=def_scheduler,
            interactive=True
        )

    with gr.Column():
        # Size Sliders
        width = gr.Slider(
            label="Width",
            minimum=64,
            maximum=2048,
            value=def_width,
            step=8
        )
        height = gr.Slider(
            label="Height",
            minimum=64,
            maximum=2048,
            value=def_height,
            step=8
        )

    with gr.Row():
        # Prediction mode
        predict = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=def_predict,
            interactive=True
        )
    with gr.Row():
        # Boolean options
        flash_attn = gr.Checkbox(
            label="Flash Attention",
            value=def_flash_attn
        )
        diffusion_conv_direct = gr.Checkbox(
            label="Conv2D Direct for diffusion",
            value=def_diffusion_conv_direct
        )
        vae_conv_direct = gr.Checkbox(
            label="Conv2D Direct for VAE",
            value=def_vae_conv_direct
        )

    # Folders options
    folders_opt_components = create_folders_opt_ui()

    ckpt_dir_txt = folders_opt_components['ckpt_dir_txt']
    unet_dir_txt = folders_opt_components['unet_dir_txt']
    vae_dir_txt = folders_opt_components['vae_dir_txt']
    clip_dir_txt = folders_opt_components['clip_dir_txt']
    emb_dir_txt = folders_opt_components['emb_dir_txt']
    lora_dir_txt = folders_opt_components['lora_dir_txt']
    taesd_dir_txt = folders_opt_components['taesd_dir_txt']
    phtmkr_dir_txt = folders_opt_components['phtmkr_dir_txt']
    upscl_dir_txt = folders_opt_components['upscl_dir_txt']
    cnnet_dir_txt = folders_opt_components['cnnet_dir_txt']
    txt2img_dir_txt = folders_opt_components['txt2img_dir_txt']
    img2img_dir_txt = folders_opt_components['img2img_dir_txt']

    # Set Defaults and Restore Defaults Buttons
    with gr.Row():
        set_btn = gr.Button(value="Set Defaults")
        set_btn.click(
            set_defaults,
            inputs=[ckpt_model, ckpt_vae, unet_model, unet_vae,
                    clip_g, clip_l, t5xxl, model_type, sampling,
                    steps, scheduler, width, height, predict,
                    flash_attn, diffusion_conv_direct, vae_conv_direct,
                    ckpt_dir_txt, unet_dir_txt, vae_dir_txt,
                    clip_dir_txt,
                    emb_dir_txt, lora_dir_txt,
                    taesd_dir_txt, phtmkr_dir_txt,
                    upscl_dir_txt, cnnet_dir_txt,
                    txt2img_dir_txt, img2img_dir_txt],
            outputs=[]
        )
        restore_btn = gr.Button(value="Restore Defaults")
        restore_btn.click(
            rst_def,
            inputs=[],
            outputs=[]
        )

"""sd.cpp-webui - Options UI"""

import gradio as gr

from modules.config import (
    set_defaults, rst_def, sd_dir, vae_dir, flux_dir, clip_l_dir,
    t5xxl_dir, def_sd, def_sd_vae, def_flux, def_flux_vae,
    def_clip_l, def_t5xxl, def_sampling, def_steps, def_scheduler,
    def_width, def_height, def_predict
)
from modules.loader import (
    get_models
)
from modules.ui import (
    create_folders_opt_ui,
)

SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
            "dpm++2mv2", "ipndm", "ipndm_v", "lcm"]
SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits"]
PREDICTION = ["Default", "eps", "v", "flow"]
RELOAD_SYMBOL = '\U0001f504'


with gr.Blocks() as options_block:
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                sd_model = gr.Dropdown(
                    label="Stable Diffusion Model",
                    choices=get_models(sd_dir),
                    scale=7,
                    value=def_sd,
                    interactive=True
                )
            with gr.Row():
                reload_sd_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_sd_model = gr.ClearButton(
                    sd_model, scale=1
                )
        with gr.Column():
            with gr.Row():
                sd_vae = gr.Dropdown(
                    label="Stable Diffusion VAE",
                    choices=get_models(vae_dir),
                    scale=7, value=def_sd_vae,
                    interactive=True
                )
            with gr.Row():
                reload_vae_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_vae = gr.ClearButton(
                    sd_vae, scale=1
                )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                flux_model = gr.Dropdown(
                    label="Flux Model",
                    choices=get_models(flux_dir),
                    scale=7,
                    value=def_flux,
                    interactive=True
                )
            with gr.Row():
                reload_flux_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_flux_model = gr.ClearButton(
                    flux_model, scale=1
                )
        with gr.Column():
            with gr.Row():
                flux_vae = gr.Dropdown(
                    label="Flux VAE",
                    choices=get_models(vae_dir),
                    scale=7, value=def_flux_vae,
                    interactive=True
                )
            with gr.Row():
                reload_vae_btn = gr.Button(
                    value=RELOAD_SYMBOL, scale=1
                )
                clear_flux_vae = gr.ClearButton(
                    flux_vae, scale=1
                )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                clip_l = gr.Dropdown(
                    label="clip_l",
                    choices=get_models(clip_l_dir),
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
                    choices=get_models(t5xxl_dir),
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
        # Schedule Dropdown
        schedule = gr.Dropdown(
            label="Schedule",
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

    # Folders options
    folders_opt_components = create_folders_opt_ui()

    sd_dir_txt = folders_opt_components['sd_dir_txt']
    flux_dir_txt = folders_opt_components['flux_dir_txt']
    vae_dir_txt = folders_opt_components['vae_dir_txt']
    clip_l_dir_txt = folders_opt_components['clip_l_dir_txt']
    t5xxl_dir_txt = folders_opt_components['t5xxl_dir_txt']
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
            inputs=[sd_model, sd_vae, flux_model, flux_vae,
                    clip_l, t5xxl, sampling, steps, schedule,
                    width, height, predict,
                    sd_dir_txt, flux_dir_txt, vae_dir_txt,
                    clip_l_dir_txt, t5xxl_dir_txt,
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

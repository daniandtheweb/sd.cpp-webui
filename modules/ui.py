"""sd.cpp-webui - Reusable UI module"""

import os

import gradio as gr

from modules.config import (
    set_defaults, rst_def, get_prompts, reload_prompts, save_prompts,
    delete_prompts, load_prompts, sd_dir, vae_dir, flux_dir, clip_l_dir,
    t5xxl_dir, emb_dir, lora_dir, taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir,
    txt2img_dir, img2img_dir, def_sd, def_sd_vae, def_flux, def_flux_vae,
    def_clip_l, def_t5xxl, def_sampling, def_steps, def_scheduler, def_width,
    def_height, def_predict
)
from modules.loader import (
    get_models, reload_models
)

SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
            "dpm++2mv2", "ipndm", "ipndm_v", "lcm"]
SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits"]
PREDICTION = ["Default", "eps", "v", "flow"]
RELOAD_SYMBOL = '\U0001f504'


def create_model_sel_ui():
    """Create the model selection UI"""
    # Dictionary to hold UI components
    model_components = {}

    sd_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    flux_dir_txt = gr.Textbox(value=flux_dir, visible=False)
    clip_l_dir_txt = gr.Textbox(value=clip_l_dir, visible=False)
    t5xxl_dir_txt = gr.Textbox(value=t5xxl_dir, visible=False)

    # Model & VAE Selection
    with gr.Row():
        with gr.Tab("Stable Diffusion") as model_components['sd_tab']:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['sd_model'] = gr.Dropdown(
                            label="Stable Diffusion Model",
                            choices=get_models(sd_dir),
                            scale=7,
                            value=def_sd,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_sd_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_sd_btn'].click(
                            reload_models,
                            inputs=[sd_dir_txt],
                            outputs=[model_components['sd_model']]
                        )

                        model_components['clear_sd_model'] = gr.ClearButton(
                            model_components['sd_model'],
                            scale=1
                        )
                with gr.Column():
                    with gr.Row():
                        model_components['sd_vae'] = gr.Dropdown(
                            label="Stable Diffusion VAE",
                            choices=get_models(vae_dir),
                            scale=7,
                            value=def_sd_vae,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_vae_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_vae_btn'].click(
                            reload_models,
                            inputs=[vae_dir_txt],
                            outputs=[model_components['sd_vae']]
                        )

                        model_components['clear_vae'] = gr.ClearButton(
                            model_components['sd_vae'],
                            scale=1
                        )

        with gr.Tab("Flux") as model_components['flux_tab']:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['flux_model'] = gr.Dropdown(
                            label="Flux Model",
                            choices=get_models(flux_dir),
                            scale=7,
                            value=def_flux,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_flux_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_flux_btn'].click(
                            reload_models,
                            inputs=[flux_dir_txt],
                            outputs=[model_components['flux_model']]
                        )

                        model_components['clear_flux_model'] = gr.ClearButton(
                            model_components['flux_model'],
                            scale=1
                        )
                with gr.Column():
                    with gr.Row():
                        model_components['flux_vae'] = gr.Dropdown(
                            label="Flux VAE",
                            choices=get_models(vae_dir),
                            scale=7,
                            value=def_flux_vae,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_flux_vae_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_flux_vae_btn'].click(
                            reload_models,
                            inputs=[vae_dir_txt],
                            outputs=[model_components['sd_vae']]
                        )

                        model_components['clear_flux_vae'] = gr.ClearButton(
                            model_components['flux_vae'],
                            scale=1
                        )

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['clip_l'] = gr.Dropdown(
                            label="clip_l",
                            choices=get_models(clip_l_dir),
                            scale=7,
                            value=def_clip_l,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_clip_l_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_clip_l_btn'].click(
                            reload_models,
                            inputs=[clip_l_dir_txt],
                            outputs=[model_components['clip_l']]
                        )

                        model_components['clear_clip_l'] = gr.ClearButton(
                            model_components['clip_l'],
                            scale=1
                        )
                with gr.Column():
                    with gr.Row():
                        model_components['t5xxl'] = gr.Dropdown(
                            label="t5xxl",
                            choices=get_models(t5xxl_dir),
                            scale=7, value=def_t5xxl,
                            interactive=True
                        )
                    with gr.Row():
                        model_components['reload_t5xxl_btn'] = gr.Button(
                            value=RELOAD_SYMBOL,
                            scale=1
                        )
                        model_components['reload_t5xxl_btn'].click(
                            reload_models,
                            inputs=[t5xxl_dir_txt],
                            outputs=[model_components['t5xxl']]
                        )

                        model_components['clear_t5xxl'] = gr.ClearButton(
                            model_components['t5xxl'],
                            scale=1
                        )

    # Return the dictionary with all UI components
    return model_components


def create_prompts_ui():
    """Create the prompts UI"""
    # Dictionary to hold UI components
    prompts_components = {}

    # Prompts
    with gr.Row():
        with gr.Accordion(
            label="Saved prompts", open=False
        ):
            with gr.Column():
                prompts_components['saved_prompts'] = gr.Dropdown(
                    label="Prompts",
                    choices=get_prompts(),
                    interactive=True,
                    allow_custom_value=True
                )
            with gr.Column():
                with gr.Row():
                    prompts_components['load_prompt_btn'] = gr.Button(
                        value="Load prompt", size="lg"
                    )
                    prompts_components['reload_prompts_btn'] = gr.Button(
                        value=RELOAD_SYMBOL
                    )
                with gr.Row():
                    prompts_components['save_prompt_btn'] = gr.Button(
                        value="Save prompt", size="lg"
                    )
                    prompts_components['del_prompt_btn'] = gr.Button(
                        value="Delete prompt", size="lg"
                    )
    with gr.Row():
        prompts_components['pprompt'] = gr.Textbox(
            placeholder="Positive prompt",
            label="Positive Prompt",
            lines=3,
            show_copy_button=True
        )
    with gr.Row():
        prompts_components['nprompt'] = gr.Textbox(
            placeholder="Negative prompt",
            label="Negative Prompt",
            lines=3,
            show_copy_button=True
        )

    # Return the dictionary with all UI components
    return prompts_components


def create_settings_ui():
    """Create settings UI"""
    # Dictionary to hold UI components
    settings_components = {}

    with gr.Row():
        with gr.Column(scale=1):
            settings_components['sampling'] = gr.Dropdown(
                label="Sampling method",
                choices=SAMPLERS,
                value=def_sampling,
                interactive=True
            )
        with gr.Column(scale=1):
            settings_components['steps'] = gr.Slider(
                label="Steps",
                minimum=1,
                maximum=99,
                value=def_steps,
                step=1
            )
    with gr.Row():
        settings_components['schedule'] = gr.Dropdown(
            label="Schedule",
            choices=SCHEDULERS,
            value=def_scheduler,
            interactive=True
        )
    with gr.Row():
        with gr.Column():
            settings_components['width'] = gr.Slider(
                label="Width",
                minimum=64,
                maximum=2048,
                value=def_width,
                step=64
            )
            settings_components['height'] = gr.Slider(
                label="Height",
                minimum=64,
                maximum=2048,
                value=def_height,
                step=64
            )
        settings_components['batch_count'] = gr.Slider(
            label="Batch count",
            minimum=1,
            maximum=99,
            value=1,
            step=1
        )
    settings_components['cfg'] = gr.Slider(
        label="CFG Scale",
        minimum=1,
        maximum=30,
        value=7.0,
        step=0.1,
        interactive=True
    )

    # Return the dictionary with all UI components
    return settings_components


def create_cnnet_ui():
    """Create the ControlNet UI"""
    # Dictionary to hold UI components
    cnnet_components = {}

    # ControlNet
    with gr.Accordion(
        label="ControlNet", open=False
    ):
        cnnet_components['cnnet'] = gr.Dropdown(
            label="ControlNet",
            choices=get_models(cnnet_dir),
            interactive=True
        )
        cnnet_components['reload_cnnet_btn'] = gr.Button(value=RELOAD_SYMBOL)
        cnnet_components['clear_cnnet'] = gr.ClearButton(
            cnnet_components['cnnet']
        )
        cnnet_components['control_img'] = gr.Image(
            sources="upload", type="filepath"
        )
        cnnet_components['control_strength'] = gr.Slider(
            label="ControlNet strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.9)
        cnnet_components['cnnet_cpu'] = gr.Checkbox(label="ControlNet on CPU")
        cnnet_components['canny'] = gr.Checkbox(label="Canny (edge detection)")

    # Return the dictionary with all UI components
    return cnnet_components


def create_extras_ui():
    """Create the extras UI"""
    # Dictionary to hold UI components
    extras_components = {}

    # Extra Settings
    with gr.Accordion(
        label="Extra", open=False
    ):
        extras_components['threads'] = gr.Number(
            label="Threads",
            minimum=0,
            maximum=os.cpu_count(),
            value=0
        )
        extras_components['vae_tiling'] = gr.Checkbox(label="VAE Tiling")
        extras_components['vae_cpu'] = gr.Checkbox(label="VAE on CPU")
        extras_components['rng'] = gr.Dropdown(
            label="RNG",
            choices=["std_default", "cuda"],
            value="cuda"
        )
        extras_components['predict'] = gr.Dropdown(
            label="Prediction",
            choices=PREDICTION,
            value=def_predict
        )
        extras_components['output'] = gr.Textbox(
            label="Output Name (optional)", value=""
        )
        extras_components['color'] = gr.Checkbox(
            label="Color", value=True
        )
        extras_components['verbose'] = gr.Checkbox(label="Verbose")

    # Return the dictionary with all UI components
    return extras_components


def create_folders_opt_ui():
    """Create the folder options UI"""
    # Dictionary to hold UI components
    folders_opt_components = {}

    with gr.Row():
        # Folders Accordion
        with gr.Accordion(
            label="Folders", open=False
        ):
            folders_opt_components['sd_dir_txt'] = gr.Textbox(
                label="Stable Diffusion folder",
                value=sd_dir,
                interactive=True
            )
            folders_opt_components['flux_dir_txt'] = gr.Textbox(
                label="Flux folder",
                value=flux_dir,
                interactive=True
            )
            folders_opt_components['vae_dir_txt'] = gr.Textbox(
                label="VAE folder",
                value=vae_dir,
                interactive=True
            )
            folders_opt_components['clip_l_dir_txt'] = gr.Textbox(
                label="clip_l folder",
                value=clip_l_dir,
                interactive=True
            )
            folders_opt_components['t5xxl_dir_txt'] = gr.Textbox(
                label="t5xxl folder",
                value=t5xxl_dir,
                interactive=True
            )
            folders_opt_components['emb_dir_txt'] = gr.Textbox(
                label="Embeddings folder",
                value=emb_dir,
                interactive=True
            )
            folders_opt_components['lora_dir_txt'] = gr.Textbox(
                label="Lora folder",
                value=lora_dir,
                interactive=True
            )
            folders_opt_components['taesd_dir_txt'] = gr.Textbox(
                label="TAESD folder",
                value=taesd_dir,
                interactive=True
            )
            folders_opt_components['phtmkr_dir_txt'] = gr.Textbox(
                label="PhotoMaker folder",
                value=phtmkr_dir,
                interactive=True
            )
            folders_opt_components['upscl_dir_txt'] = gr.Textbox(
                label="Upscaler folder",
                value=upscl_dir,
                interactive=True
            )
            folders_opt_components['cnnet_dir_txt'] = gr.Textbox(
                label="ControlNet folder",
                value=cnnet_dir,
                interactive=True
            )
            folders_opt_components['txt2img_dir_txt'] = gr.Textbox(
                label="txt2img outputs folder",
                value=txt2img_dir,
                interactive=True
            )
            folders_opt_components['img2img_dir_txt'] = gr.Textbox(
                label="img2img outputs folder",
                value=img2img_dir,
                interactive=True
            )

    # Return the dictionary with all UI components
    return folders_opt_components

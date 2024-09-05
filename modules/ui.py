"""sd.cpp-webui - Reusable UI module"""

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
    get_models
)

RELOAD_SYMBOL = '\U0001f504'

def create_model_ui():
    """Create the model selection UI"""
    # Dictionary to hold UI components
    model_components = {}

    # Model & VAE Selection
    with gr.Row():
        with gr.Tab("Stable Diffusion") as model_components['sd_tab']:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['sd_model'] = gr.Dropdown(label="Stable Diffusion Model",
                                                                   choices=get_models(sd_dir),
                                                                   scale=7, value=def_sd,
                                                                   interactive=True)
                    with gr.Row():
                        model_components['reload_sd_btn'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_sd_model'] = gr.ClearButton(model_components['sd_model'], scale=1)
                with gr.Column():
                    with gr.Row():
                        model_components['sd_vae'] = gr.Dropdown(label="Stable Diffusion VAE",
                                                                 choices=get_models(vae_dir),
                                                                 scale=7, value=def_sd_vae,
                                                                 interactive=True)
                    with gr.Row():
                        model_components['reload_vae_btn'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_vae'] = gr.ClearButton(model_components['sd_vae'], scale=1)

        with gr.Tab("Flux") as model_components['flux_tab']:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['flux_model'] = gr.Dropdown(label="Flux Model",
                                                                     choices=get_models(flux_dir),
                                                                     scale=7, value=def_flux,
                                                                     interactive=True)
                    with gr.Row():
                        model_components['reload_flux_btn'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_flux_model'] = gr.ClearButton(model_components['flux_model'], scale=1)
                with gr.Column():
                    with gr.Row():
                        model_components['flux_vae'] = gr.Dropdown(label="Flux VAE",
                                                                   choices=get_models(vae_dir),
                                                                   scale=7, value=def_flux_vae,
                                                                   interactive=True)
                    with gr.Row():
                        model_components['reload_vae_btn_2'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_flux_vae'] = gr.ClearButton(model_components['flux_vae'], scale=1)

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_components['clip_l'] = gr.Dropdown(label="clip_l",
                                                                 choices=get_models(clip_l_dir),
                                                                 scale=7, value=def_clip_l,
                                                                 interactive=True)
                    with gr.Row():
                        model_components['reload_clip_l_btn'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_clip_l'] = gr.ClearButton(model_components['clip_l'], scale=1)
                with gr.Column():
                    with gr.Row():
                        model_components['t5xxl'] = gr.Dropdown(label="t5xxl",
                                                                choices=get_models(t5xxl_dir),
                                                                scale=7, value=def_t5xxl,
                                                                interactive=True)
                    with gr.Row():
                        model_components['reload_t5xxl_btn'] = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        model_components['clear_t5xxl'] = gr.ClearButton(model_components['t5xxl'], scale=1)

    # Return the dictionary with all UI components
    return model_components

"""sd.cpp-webui - Image edit UI"""

import gradio as gr

from modules.sdcpp import imgedit
from modules.utils.utility import random_seed
from modules.utils.ui_handler import refresh_all_options
from modules.shared_instance import (
    config, subprocess_manager
)
from modules.loader import (
    get_models, reload_models
)
from modules.ui.constants import RELOAD_SYMBOL, RANDOM_SYMBOL
from modules.ui.models import create_unet_model_sel_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.eta import create_eta_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.advanced_settings import create_extras_ui
from modules.ui.environment import create_env_ui
from modules.ui.experimental import create_experimental_ui


with gr.Blocks() as imgedit_block:
    inputs_map = {}
    diffusion_mode = gr.Number(value=1, visible=False)
    # Directory Textboxes
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)

    # Title
    imgedit_title = gr.Markdown("# Image Edit")

    # Model & VAE Selection
    model_ui = create_unet_model_sel_ui()
    inputs_map.update(model_ui)
    inputs_map['in_diffusion_mode'] = diffusion_mode

    # Model Type Selection
    quant_ui = create_quant_ui()
    inputs_map.update(quant_ui)

    # Extra Networks Selection
    with gr.Accordion(label="Extra Networks", open=False):
        with gr.Row():
            with gr.Group():
                with gr.Row():
                    taesd_model = gr.Dropdown(
                        label="TAESD",
                        choices=get_models(config.get('taesd_dir')),
                        value="",
                        allow_custom_value=True,
                        interactive=True
                    )
                with gr.Row():
                    reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL)
                    gr.ClearButton(taesd_model)
                inputs_map['in_taesd'] = taesd_model

        # PhotoMaker
        photomaker_ui = create_photomaker_ui()
        inputs_map.update(photomaker_ui)

    # Prompts
    prompts_ui = create_prompts_ui(nprompt_support = False)
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            generation_settings_ui = create_generation_settings_ui(unet_mode=True)
            inputs_map.update(generation_settings_ui)

            with gr.Row():
                with gr.Group():
                    seed = gr.Number(
                        label="Seed",
                        minimum=-1,
                        maximum=10**16,
                        value=-1,
                        scale=5
                    )
                    random_seed_btn = gr.Button(
                        value=RANDOM_SYMBOL, scale=1
                    )
                    inputs_map['in_seed'] = seed

            clip_skip = gr.Slider(
                label="CLIP skip",
                minimum=-1,
                maximum=12,
                value=-1,
                step=1
            )
            inputs_map['in_clip_skip'] = clip_skip

            # Upscale
            upscl_ui = create_upscl_ui()
            inputs_map.update(upscl_ui)

            # ControlNet
            cnnet_ui = create_cnnet_ui()
            inputs_map.update(cnnet_ui)

            # ETA
            eta_ui = create_eta_ui()
            inputs_map.update(eta_ui)

            # VAE Tiling
            vae_tiling_ui = create_vae_tiling_ui()
            inputs_map.update(vae_tiling_ui)

            # Extra Settings
            extras_ui = create_extras_ui()
            inputs_map.update(extras_ui)

            # Environment Variables
            env_ui = create_env_ui()
            inputs_map.update(env_ui)

            # Experimental
            experimental_ui = create_experimental_ui()
            inputs_map.update(experimental_ui)

            with gr.Row():
                refresh_opt = gr.Button(
                    value="Refresh sd options"
                )

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                ref_img_imgedit = gr.Image(
                    sources="upload", type="filepath"
                )
                inputs_map['in_ref_img'] = ref_img_imgedit
            with gr.Group():
                with gr.Row():
                    gen_btn = gr.Button(
                        value="Generate", size="lg",
                        variant="primary"
                    )
                    kill_btn = gr.Button(
                        value="Stop", size="lg",
                        variant="stop"
                    )
                with gr.Row():
                    progress_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False,
                        visible=False,
                        label="Progress",
                        show_reset_button=False
                    )
                with gr.Row():
                    progress_textbox = gr.Textbox(
                        label="Status:",
                        visible=False,
                        interactive=False
                    )
                with gr.Row():
                    img_final = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        columns=[3],
                        rows=[1],
                        object_fit="contain",
                        height="auto"
                    )
                with gr.Row():
                    stats = gr.Textbox(
                        label="Statistics:",
                        show_label=True,
                        value="",
                        interactive=False
                    )
                with gr.Row():
                    command = gr.Textbox(
                        label="stable-diffusion.cpp command:",
                        show_label=True,
                        value="",
                        interactive=False,
                        show_copy_button=True,
                    )

    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[key] for key in ordered_keys]

    def imgedit_wrapper(*args):
        """
        Accepts all UI inputs, zips them with keys, and calls the
        main imgedit function.
        """
        # This line programmatically creates the dictionary.
        params = dict(zip(ordered_keys, args))
        yield from imgedit(params)

    # Generate
    gen_btn.click(
        imgedit_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_slider, progress_textbox, stats, img_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )
    reload_taesd_btn.click(
        reload_models, inputs=[taesd_dir_txt], outputs=[taesd_model]
    )
    random_seed_btn.click(
        random_seed, inputs=[], outputs=[seed]
    )
    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            experimental_ui['in_preview_mode'], extras_ui['in_predict']
        ]
    )

    width_imgedit = generation_settings_ui['in_width']
    height_imgedit = generation_settings_ui['in_height']

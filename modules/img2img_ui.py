"""sd.cpp-webui - Image to image UI"""

import gradio as gr
from functools import partial

from modules.sdcpp import img2img
from modules.utils.utility import random_seed
from modules.utils.ui_handler import (
    ckpt_tab_switch, unet_tab_switch, update_interactivity,
    refresh_all_options
)
from modules.shared_instance import (
    config, subprocess_manager
)
from modules.loader import (
    get_models, reload_models
)
from modules.ui.constants import RELOAD_SYMBOL, RANDOM_SYMBOL
from modules.ui.models import create_img_model_sel_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.chroma import create_chroma_ui
from modules.ui.timestep_shift import create_timestep_shift_ui
from modules.ui.eta import create_eta_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.advanced_settings import create_extras_ui
from modules.ui.environment import create_env_ui
from modules.ui.experimental import create_experimental_ui


with gr.Blocks()as img2img_block:
    inputs_map = {}
    # Directory Textboxes
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)

    # Title
    img2img_title = gr.Markdown("# Image to Image")

    # Model & VAE Selection
    model_ui = create_img_model_sel_ui()
    inputs_map.update(model_ui['inputs'])

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

        phtmkr_ui = create_photomaker_ui()
        inputs_map.update(phtmkr_ui)

    # Prompts
    prompts_ui = create_prompts_ui()
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            generation_settings_ui = create_generation_settings_ui()
            inputs_map.update(generation_settings_ui)

            with gr.Row():
                img_cfg_bool = gr.Checkbox(
                    label="Enable Image CFG",
                    value=False
                )
                img_cfg = gr.Slider(
                    label="Image CFG (inpaint or instruct-pix2pix models)",
                    minimum=1,
                    maximum=30,
                    value=7.0,
                    step=0.1,
                    interactive=False
                )
                inputs_map['in_img_cfg'] = img_cfg

                cfg_comp = [img_cfg]

            with gr.Row():
                strenght = gr.Slider(
                    label="Noise strenght",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.75
                )
                inputs_map['in_strenght'] = strenght

            with gr.Row():
                style_ratio_bool = gr.Checkbox(
                    label="Enable style-ratio",
                    value=False
                )
                style_ratio = gr.Slider(
                    label="Style ratio",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=20,
                    interactive=False
                )
                inputs_map['in_style_ratio_bool'] = style_ratio_bool
                inputs_map['in_style_ratio'] = style_ratio

                style_comp = [style_ratio]

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

            # Chroma
            chroma_ui = create_chroma_ui()
            inputs_map.update(chroma_ui)

            # Timestep shift
            timestep_shift_ui = create_timestep_shift_ui()
            inputs_map.update(timestep_shift_ui)

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
                img_inp_img2img = gr.Image(
                    sources="upload", type="filepath"
                )
                inputs_map['in_img_inp'] = img_inp_img2img
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
                        label="Progress"
                    )
                with gr.Row():
                    progress_textbox = gr.Textbox(
                        label="Progress:",
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

    def img2img_wrapper(*args):
        """
        Accepts all UI inputs, zips them with keys, and calls the
        main img2img function.
        """
        # This line programmatically creates the dictionary.
        params = dict(zip(ordered_keys, args))
        yield from img2img(params)

    # Generate
    gen_btn.click(
        img2img_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_slider, progress_textbox, stats, img_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    model_ui['components']['ckpt_tab'].select(
        ckpt_tab_switch,
        inputs=[
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            model_ui['inputs']['in_qwen2vl'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ],
        outputs=[
            model_ui['inputs']['in_diffusion_mode'],
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            model_ui['inputs']['in_qwen2vl'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ]
    )
    model_ui['components']['unet_tab'].select(
        unet_tab_switch,
        inputs=[
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_ckpt_vae'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ],
        outputs=[
            model_ui['inputs']['in_diffusion_mode'],
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            model_ui['inputs']['in_qwen2vl'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ]
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

    img_cfg_bool.change(
        partial(update_interactivity, len(cfg_comp)),
        inputs=img_cfg_bool,
        outputs=cfg_comp
    )

    style_ratio_bool.change(
        partial(update_interactivity, len(style_comp)),
        inputs=style_ratio_bool,
        outputs=style_comp
    )

    pprompt_img2img = prompts_ui['in_pprompt']
    nprompt_img2img = prompts_ui['in_nprompt']
    width_img2img = generation_settings_ui['in_width']
    height_img2img = generation_settings_ui['in_height']
    steps_img2img = generation_settings_ui['in_steps']
    sampling_img2img = generation_settings_ui['in_sampling']
    scheduler_img2img = generation_settings_ui['in_scheduler']
    cfg_img2img = generation_settings_ui['in_cfg']
    seed_img2img = inputs_map['in_seed']

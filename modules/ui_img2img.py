"""sd.cpp-webui - Image to image UI"""

import gradio as gr

from modules.sdcpp import img2img
from modules.utility import (
    subprocess_manager, random_seed, ckpt_tab_switch, unet_tab_switch,
    SDOptionsCache
)
from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.ui import (
    create_img_model_sel_ui, create_quant_ui, create_prompts_ui,
    create_experimental_ui, create_cnnet_ui, create_chroma_ui,
    create_extras_ui, create_env_ui, create_settings_ui,
    RELOAD_SYMBOL, RANDOM_SYMBOL
)


sd_options = SDOptionsCache()


def refresh_all_options():
    sd_options.refresh()
    return [
        gr.update(choices=sd_options.get_opt("samplers")),
        gr.update(choices=sd_options.get_opt("schedulers")),
        gr.update(choices=["none"] + sd_options.get_opt("previews")),
        gr.update(choices=["Default"] + sd_options.get_opt("prediction"))
    ]


with gr.Blocks()as img2img_block:
    inputs_map = {}
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=config.get('emb_dir'), visible=False)
    lora_dir_txt = gr.Textbox(value=config.get('lora_dir'), visible=False)
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)
    phtmkr_dir_txt = gr.Textbox(value=config.get('phtmkr_dir'), visible=False)
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

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
        taesd_title = gr.Markdown("## TAESD")
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

        with gr.Row():
            phtmkr_title = gr.Markdown("## PhotoMaker")
        with gr.Row():
            phtmkr_model = gr.Dropdown(
                label="PhotoMaker",
                choices=get_models(config.get('phtmkr_dir')),
                value="",
                allow_custom_value=True,
                interactive=True
            )
        with gr.Row():
            reload_phtmkr_btn = gr.Button(value=RELOAD_SYMBOL)
            gr.ClearButton(phtmkr_model)
        with gr.Row():
            phtmkr_in = gr.Textbox(
                label="PhotoMaker images directory",
                value="",
                interactive=True
            )
        with gr.Row():
            gr.ClearButton(phtmkr_in)
        with gr.Row():
            phtmkr_nrml = gr.Checkbox(
                label="Normalize PhotoMaker input", value=False
            )
        inputs_map['in_phtmkr'] = phtmkr_model
        inputs_map['in_phtmkr_in'] = phtmkr_in
        inputs_map['in_phtmkr_nrml'] = phtmkr_nrml

    # Prompts
    prompts_ui = create_prompts_ui()
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            settings_ui = create_settings_ui()
            inputs_map.update(settings_ui)

            with gr.Row():
                img_cfg_btn = gr.Checkbox(label="Enable Image CFG")
                img_cfg = gr.Slider(
                    label="Image CFG (inpaint or instruct-pix2pix models)",
                    minimum=1,
                    maximum=30,
                    value=7.0,
                    step=0.1,
                    interactive=True
                )
                inputs_map['in_img_cfg'] = img_cfg

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
                style_ratio_btn = gr.Checkbox(label="Enable style-ratio")
                style_ratio = gr.Slider(
                    label="Style ratio",
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=20
                )
                inputs_map['in_style_ratio_btn'] = style_ratio_btn
                inputs_map['in_style_ratio'] = style_ratio

            with gr.Row():
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
            with gr.Accordion(
                label="Upscale", open=False
            ):
                upscl = gr.Dropdown(
                    label="Upscaler",
                    choices=get_models(config.get('upscl_dir')),
                    value="",
                    allow_custom_value=True,
                    interactive=True
                )
                reload_upscl_btn = gr.Button(value=RELOAD_SYMBOL)
                gr.ClearButton(upscl)
                upscl_rep = gr.Slider(
                    label="Upscaler repeats",
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=0.1
                )
                inputs_map['in_upscl'] = upscl
                inputs_map['in_upscl_rep'] = upscl_rep

            # ControlNet
            cnnet_ui = create_cnnet_ui()
            inputs_map.update(cnnet_ui)

            # Chroma
            chroma_ui = create_chroma_ui()
            inputs_map.update(chroma_ui)

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
                img_inp = gr.Image(
                    sources="upload", type="filepath"
                )
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
            settings_ui['in_guidance_btn'],
            settings_ui['in_guidance']
        ],
        outputs=[
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            settings_ui['in_guidance_btn'],
            settings_ui['in_guidance']
        ]
    )
    model_ui['components']['unet_tab'].select(
        unet_tab_switch,
        inputs=[
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_ckpt_vae'],
            settings_ui['in_guidance_btn'],
            settings_ui['in_guidance']
        ],
        outputs=[
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            settings_ui['in_guidance_btn'],
            settings_ui['in_guidance']
        ]
    )
    reload_taesd_btn.click(
        reload_models, inputs=[taesd_dir_txt], outputs=[taesd_model]
    )
    reload_phtmkr_btn.click(
        reload_models, inputs=[phtmkr_dir_txt], outputs=[phtmkr_model]
    )
    reload_upscl_btn.click(
        reload_models, inputs=[upscl_dir_txt], outputs=[upscl]
    )
    random_seed_btn.click(
        random_seed, inputs=[], outputs=[seed]
    )
    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            settings_ui['in_sampling'], settings_ui['in_scheduler'],
            experimental_ui['in_preview_mode'], experimental_ui['in_predict']
        ]
    )

    pprompt_img2img = prompts_ui['in_pprompt']
    nprompt_img2img = prompts_ui['in_nprompt']
    width_img2img = settings_ui['in_width']
    height_img2img = settings_ui['in_height']
    steps_img2img = settings_ui['in_steps']
    sampling_img2img = settings_ui['in_sampling']
    scheduler_img2img = settings_ui['in_scheduler']
    cfg_img2img = settings_ui['in_cfg']
    seed_img2img = inputs_map['in_seed']

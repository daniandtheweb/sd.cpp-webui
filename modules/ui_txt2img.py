"""sd.cpp-webui - Text to image UI"""

import gradio as gr

from modules.sdcpp import txt2img
from modules.utility import (
    subprocess_manager, random_seed, ckpt_tab_switch, unet_tab_switch
)
from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.ui import (
    create_img_model_sel_ui, create_prompts_ui,
    create_cnnet_ui, create_extras_ui, create_settings_ui,
    QUANTS, RELOAD_SYMBOL, RANDOM_SYMBOL
)


TXT2IMG_KEYS = [
    'in_ckpt_model', 'in_ckpt_vae', 'in_unet_model', 'in_unet_vae',
    'in_clip_g', 'in_clip_l', 'in_t5xxl', 'in_model_type', 'in_taesd',
    'in_phtmkr', 'in_phtmkr_in', 'in_phtmkr_nrml', 'in_upscl',
    'in_upscl_rep', 'in_cnnet', 'in_control_img', 'in_control_strength',
    'in_ppromt', 'in_nprompt', 'in_sampling', 'in_steps', 'in_scheduler',
    'in_width', 'in_height', 'in_batch_count', 'in_cfg', 'in_guidance_btn',
    'in_guidance', 'in_seed', 'in_clip_skip', 'in_threads',
    'in_offload_to_cpu', 'in_vae_tiling', 'in_vae_cpu', 'in_clip_cpu',
    'in_cnnet_cpu', 'in_canny', 'in_rng', 'in_predict', 'in_output',
    'in_color', 'in_flash_attn', 'in_diffusion_conv_direct',
    'in_vae_conv_direct', 'in_verbose'
]


def txt2img_wrapper(*args):
    """
    Accepts all UI inputs, zips them with keys, and calls the
    main txt2img function.
    """
    # This line programmatically creates the dictionary.
    params = dict(zip(TXT2IMG_KEYS, args))
    yield from txt2img(params)


with gr.Blocks() as txt2img_block:
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=config.get('emb_dir'), visible=False)
    lora_dir_txt = gr.Textbox(value=config.get('lora_dir'), visible=False)
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)
    phtmkr_dir_txt = gr.Textbox(value=config.get('phtmkr_dir'), visible=False)
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    # Title
    txt2img_title = gr.Markdown("# Text to Image")

    # Model & VAE Selection
    img_model_components = create_img_model_sel_ui()

    # Checkpoint Tab Components
    ckpt_tab = img_model_components['ckpt_tab']
    ckpt_model = img_model_components['ckpt_model']
    reload_ckpt_btn = img_model_components['reload_ckpt_btn']
    clear_ckpt_model = img_model_components['clear_ckpt_model']
    ckpt_vae = img_model_components['ckpt_vae']
    reload_vae_btn = img_model_components['reload_vae_btn']
    clear_vae = img_model_components['clear_vae']

    # UNET Tab Components (FLUX, Stable Diffusion 3/3.5)
    unet_tab = img_model_components['unet_tab']
    unet_model = img_model_components['unet_model']
    reload_unet_btn = img_model_components['reload_unet_btn']
    clear_unet_model = img_model_components['clear_unet_model']
    unet_vae = img_model_components['unet_vae']
    reload_unet_vae_btn = img_model_components['reload_unet_vae_btn']
    clear_unet_vae = img_model_components['clear_unet_vae']
    clip_g = img_model_components['clip_g']
    reload_clip_g_btn = img_model_components['reload_clip_g_btn']
    clear_clip_g = img_model_components['clear_clip_g']
    clip_l = img_model_components['clip_l']
    reload_clip_l_btn = img_model_components['reload_clip_l_btn']
    clear_clip_l = img_model_components['clear_clip_l']
    t5xxl = img_model_components['t5xxl']
    reload_t5xxl_btn = img_model_components['reload_t5xxl_btn']
    clear_t5xxl = img_model_components['clear_t5xxl']

    # Model Type Selection
    with gr.Row():
        model_type = gr.Dropdown(
            label="Quantization",
            choices=QUANTS,
            value=config.get('def_type'),
            interactive=True
        )

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
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
                clear_taesd = gr.ClearButton(taesd_model)
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
                clear_phtmkr = gr.ClearButton(phtmkr_model)
            with gr.Row():
                phtmkr_in = gr.Textbox(
                    label="PhotoMaker images directory",
                    value="",
                    interactive=True
                )
            with gr.Row():
                clear_phtmkr_in = gr.ClearButton(phtmkr_in)
            with gr.Row():
                phtmkr_nrml = gr.Checkbox(
                    label="Normalize PhotoMaker input", value=False
                )

    # Prompts
    prompts_components = create_prompts_ui()

    saved_prompts = prompts_components['saved_prompts']
    load_prompt_btn = prompts_components['load_prompt_btn']
    reload_prompts_btn = prompts_components['reload_prompts_btn']
    save_prompt_btn = prompts_components['save_prompt_btn']
    del_prompt_btn = prompts_components['del_prompt_btn']
    pprompt_txt2img = prompts_components['pprompt']
    nprompt_txt2img = prompts_components['nprompt']

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            settings_components = create_settings_ui()

            sampling_txt2img = settings_components['sampling']
            steps_txt2img = settings_components['steps']
            scheduler = settings_components['scheduler']
            width_txt2img = settings_components['width']
            height_txt2img = settings_components['height']
            switch_size = settings_components['switch_size']
            batch_count = settings_components['batch_count']
            cfg_txt2img = settings_components['cfg']
            guidance_btn = settings_components['guidance_btn']
            guidance = settings_components['guidance']

            with gr.Row():
                seed_txt2img = gr.Number(
                    label="Seed",
                    minimum=-1,
                    maximum=10**16,
                    value=-1,
                    scale=5
                )
                random_seed_btn = gr.Button(
                    value=RANDOM_SYMBOL, scale=1
                )
            clip_skip = gr.Slider(
                label="CLIP skip",
                minimum=-1,
                maximum=12,
                value=-1,
                step=1
            )

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
                clear_upscl = gr.ClearButton(upscl)
                upscl_rep = gr.Slider(
                    label="Upscaler repeats",
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=0.1
                )

            # ControlNet
            cnnet_components = create_cnnet_ui()

            cnnet = cnnet_components['cnnet']
            reload_cnnet_btn = cnnet_components['reload_cnnet_btn']
            clear_cnnet = cnnet_components['clear_cnnet']
            control_img = cnnet_components['control_img']
            control_strength = cnnet_components['control_strength']
            cnnet_cpu = cnnet_components['cnnet_cpu']
            canny = cnnet_components['canny']

            # Extra Settings
            extras_components = create_extras_ui()

            threads = extras_components['threads']
            offload_to_cpu = extras_components['offload_to_cpu']
            vae_tiling = extras_components['vae_tiling']
            vae_cpu = extras_components['vae_cpu']
            clip_cpu = extras_components['clip_cpu']
            rng = extras_components['rng']
            predict = extras_components['predict']
            output = extras_components['output']
            color = extras_components['color']
            flash_attn = extras_components['flash_attn']
            diffusion_conv_direct = extras_components['diffusion_conv_direct']
            vae_conv_direct = extras_components['vae_conv_direct']
            verbose = extras_components['verbose']

        # Output
        with gr.Column(scale=1):
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

    # Generate
    gen_btn.click(
        txt2img_wrapper,
        inputs=[ckpt_model, ckpt_vae, unet_model, unet_vae,
                clip_g, clip_l, t5xxl, model_type, taesd_model,
                phtmkr_model, phtmkr_in, phtmkr_nrml,
                upscl, upscl_rep, cnnet, control_img,
                control_strength, pprompt_txt2img, nprompt_txt2img,
                sampling_txt2img, steps_txt2img, scheduler, width_txt2img,
                height_txt2img, batch_count, cfg_txt2img, guidance_btn,
                guidance, seed_txt2img, clip_skip, threads, offload_to_cpu,
                vae_tiling, vae_cpu, clip_cpu, cnnet_cpu, canny, rng, predict,
                output, color, flash_attn, diffusion_conv_direct,
                vae_conv_direct, verbose],
        outputs=[command, progress_slider, progress_textbox, stats, img_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    ckpt_tab.select(
        ckpt_tab_switch,
        inputs=[unet_model, unet_vae, clip_g, clip_l, t5xxl,
                guidance_btn, guidance],
        outputs=[ckpt_model, unet_model, ckpt_vae, unet_vae, clip_g, clip_l,
                 t5xxl, guidance_btn, guidance]
    )
    unet_tab.select(
        unet_tab_switch,
        inputs=[ckpt_model, ckpt_vae, guidance_btn, guidance],
        outputs=[ckpt_model, unet_model, ckpt_vae, unet_vae, clip_g, clip_l,
                 t5xxl, guidance_btn, guidance]
    )
    reload_taesd_btn.click(
        reload_models,
        inputs=[taesd_dir_txt],
        outputs=[taesd_model]
    )
    reload_phtmkr_btn.click(
        reload_models,
        inputs=[phtmkr_dir_txt],
        outputs=[phtmkr_model]
    )
    reload_upscl_btn.click(
        reload_models,
        inputs=[upscl_dir_txt],
        outputs=[upscl]
    )
    reload_cnnet_btn.click(
        reload_models,
        inputs=[cnnet_dir_txt],
        outputs=[cnnet]
    )
    save_prompt_btn.click(
        config.add_prompt,
        inputs=[saved_prompts, pprompt_txt2img,
                nprompt_txt2img],
        outputs=[]
    )
    del_prompt_btn.click(
        config.delete_prompt,
        inputs=[saved_prompts],
        outputs=[]
    )
    reload_prompts_btn.click(
        config.get_prompts,
        inputs=[],
        outputs=[saved_prompts]
    )
    load_prompt_btn.click(
        config.get_prompt,
        inputs=[saved_prompts],
        outputs=[pprompt_txt2img, nprompt_txt2img]
    )
    random_seed_btn.click(
        random_seed,
        inputs=[],
        outputs=[seed_txt2img])

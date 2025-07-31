"""sd.cpp-webui - Image to image UI"""

import gradio as gr

from modules.sdcpp import img2img
from modules.utility import (
    subprocess_manager, random_seed, ckpt_tab_switch, unet_tab_switch
)
from modules.config import (
    reload_prompts, save_prompts, delete_prompts, load_prompts,
    emb_dir, lora_dir, taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir,
    def_type
)
from modules.loader import (
    get_models, reload_models
)
from modules.ui import (
    create_model_sel_ui, create_prompts_ui,
    create_cnnet_ui, create_extras_ui, create_settings_ui,
    QUANTS, RELOAD_SYMBOL, RANDOM_SYMBOL, SWITCH_V_SYMBOL
)


with gr.Blocks()as img2img_block:
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    phtmkr_dir_txt = gr.Textbox(value=phtmkr_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    img2img_title = gr.Markdown("# Image to Image")

    # Model & VAE Selection
    model_components = create_model_sel_ui()

    # Checkpoint Tab Components
    ckpt_tab = model_components['ckpt_tab']
    ckpt_model = model_components['ckpt_model']
    reload_ckpt_btn = model_components['reload_ckpt_btn']
    clear_ckpt_model = model_components['clear_ckpt_model']
    ckpt_vae = model_components['ckpt_vae']
    reload_vae_btn = model_components['reload_vae_btn']
    clear_vae = model_components['clear_vae']

    # UNET Tab Components
    unet_tab = model_components['unet_tab']
    unet_model = model_components['unet_model']
    reload_unet_btn = model_components['reload_unet_btn']
    clear_unet_model = model_components['clear_unet_model']
    unet_vae = model_components['unet_vae']
    reload_unet_vae_btn = model_components['reload_unet_vae_btn']
    clear_unet_vae = model_components['clear_unet_vae']
    clip_g = model_components['clip_g']
    reload_clip_g_btn = model_components['reload_clip_g_btn']
    clear_clip_g = model_components['clear_clip_g']
    clip_l = model_components['clip_l']
    reload_clip_l_btn = model_components['reload_clip_l_btn']
    clear_clip_l = model_components['clear_clip_l']
    t5xxl = model_components['t5xxl']
    reload_t5xxl_btn = model_components['reload_t5xxl_btn']
    clear_t5xxl = model_components['clear_t5xxl']

    # Model Type Selection
    with gr.Row():
        model_type = gr.Dropdown(
            label="Quantization",
            choices=QUANTS,
            value=def_type,
            interactive=True
        )

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(
            label="Extra Networks", open=False
        ):
            with gr.Row():
                taesd_title = gr.Markdown("## TAESD")
            with gr.Row():
                taesd_model = gr.Dropdown(
                    label="TAESD",
                    choices=get_models(taesd_dir),
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
                    choices=get_models(phtmkr_dir),
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
    pprompt_img2img = prompts_components['pprompt']
    nprompt_img2img = prompts_components['nprompt']

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            settings_components = create_settings_ui()

            sampling_img2img = settings_components['sampling']
            steps_img2img = settings_components['steps']
            schedule = settings_components['schedule']
            width_img2img = settings_components['width']
            height_img2img = settings_components['height']
            switch_size = settings_components['switch_size']
            batch_count = settings_components['batch_count']
            cfg_img2img = settings_components['cfg']

            strenght = gr.Slider(
                label="Noise strenght",
                minimum=0,
                maximum=1,
                step=0.01,
                value=0.75
            )
            style_ratio_btn = gr.Checkbox(label="Enable style-ratio")
            style_ratio = gr.Slider(
                label="Style ratio",
                minimum=0,
                maximum=100,
                step=1,
                value=20
            )
            with gr.Row():
                seed_img2img = gr.Number(
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
                    choices=get_models(upscl_dir),
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
            vae_tiling = extras_components['vae_tiling']
            vae_cpu = extras_components['vae_cpu']
            rng = extras_components['rng']
            predict = extras_components['predict']
            output = extras_components['output']
            color = extras_components['color']
            flash_attn = extras_components['flash_attn']
            diffusion_conv_direct = extras_components['diffusion_conv_direct']
            vae_conv_direct = extras_components['vae_conv_direct']
            verbose = extras_components['verbose']

        with gr.Column(scale=1):
            with gr.Row():
                img_inp = gr.Image(
                    sources="upload", type="filepath"
                )
            with gr.Row():
                gen_btn = gr.Button(value="Generate")
                kill_btn = gr.Button(value="Stop")
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
                command = gr.Textbox(
                    label="stable-diffusion.cpp command:",
                    show_label=True,
                    value="",
                    interactive=False,
                    show_copy_button=True,
                )

    # Generate
    gen_btn.click(
        img2img,
        inputs=[ckpt_model, ckpt_vae, unet_model, unet_vae,
                clip_g, clip_l, t5xxl, model_type, taesd_model,
                phtmkr_model, phtmkr_in, phtmkr_nrml,
                img_inp, upscl, upscl_rep, cnnet,
                control_img, control_strength, pprompt_img2img,
                nprompt_img2img, sampling_img2img, steps_img2img, schedule,
                width_img2img, height_img2img, batch_count,
                strenght, style_ratio, style_ratio_btn,
                cfg_img2img, seed_img2img, clip_skip, threads, vae_tiling,
                vae_cpu, cnnet_cpu, canny, rng, predict,
                output, color, flash_attn, diffusion_conv_direct,
                vae_conv_direct, verbose],
        outputs=[command, img_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    ckpt_tab.select(
        ckpt_tab_switch,
        inputs=[unet_model, unet_vae, clip_g, clip_l, t5xxl],
        outputs=[ckpt_model, unet_model, ckpt_vae, unet_vae, clip_g, clip_l,
                 t5xxl, pprompt_img2img, nprompt_img2img]
    )
    unet_tab.select(
        unet_tab_switch,
        inputs=[ckpt_model, ckpt_vae, nprompt_img2img],
        outputs=[ckpt_model, unet_model, ckpt_vae, unet_vae, clip_g, clip_l,
                 t5xxl, pprompt_img2img, nprompt_img2img]
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
        save_prompts,
        inputs=[saved_prompts, pprompt_img2img, nprompt_img2img],
        outputs=[]
    )
    del_prompt_btn.click(
        delete_prompts,
        inputs=[saved_prompts],
        outputs=[]
    )
    reload_prompts_btn.click(
        reload_prompts,
        inputs=[],
        outputs=[saved_prompts]
    )
    load_prompt_btn.click(
        load_prompts,
        inputs=[saved_prompts],
        outputs=[pprompt_img2img, nprompt_img2img]
    )
    random_seed_btn.click(
        random_seed,
        inputs=[],
        outputs=[seed_img2img]
    )

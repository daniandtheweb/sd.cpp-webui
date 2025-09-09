"""sd.cpp-webui - Anything to Video UI"""

import gradio as gr

from modules.sdcpp import any2video
from modules.utility import (
    subprocess_manager, random_seed
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
    create_video_model_sel_ui, create_prompts_ui,
    create_cnnet_ui, create_extras_ui, create_settings_ui,
    QUANTS, RELOAD_SYMBOL, RANDOM_SYMBOL
)


with gr.Blocks() as any2video_block:
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    phtmkr_dir_txt = gr.Textbox(value=phtmkr_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    any2video_title = gr.Markdown("# Anything to Video")

    # Model & VAE Selection
    video_model_components = create_video_model_sel_ui()

    # UNET Components (Wan 2.1/2.2)
    unet_model = video_model_components['unet_model']
    reload_unet_btn = video_model_components['reload_unet_btn']
    clear_unet_model = video_model_components['clear_unet_model']
    unet_vae = video_model_components['unet_vae']
    reload_unet_vae_btn = video_model_components['reload_unet_vae_btn']
    clear_unet_vae = video_model_components['clear_unet_vae']
    umt5_xxl = video_model_components['umt5_xxl']
    reload_umt5_xxl_btn = video_model_components['reload_umt5_xxl_btn']
    clear_umt5_xxl = video_model_components['clear_umt5_xxl']
    clip_vision_h = video_model_components['clip_vision_h']
    reload_clip_vision_h_btn = video_model_components['reload_clip_vision_h_btn']
    clear_clip_vision_h = video_model_components['clear_clip_vision_h']
    high_noise_model = video_model_components['high_noise_model']
    reload_high_noise_model = video_model_components['reload_high_noise_model']
    clear_high_noise_model = video_model_components['clear_high_noise_model']

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
        with gr.Accordion(label="Extra Networks", open=False):
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
    pprompt_any2video = prompts_components['pprompt']
    nprompt_any2video = prompts_components['nprompt']

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            settings_components = create_settings_ui()

            sampling_any2video = settings_components['sampling']
            steps_any2video = settings_components['steps']
            scheduler = settings_components['scheduler']
            width_any2video = settings_components['width']
            height_any2video = settings_components['height']
            switch_size = settings_components['switch_size']
            batch_count = settings_components['batch_count']
            cfg_any2video = settings_components['cfg']

            with gr.Row():
                frames = gr.Number(
                    label="Video Frames",
                    minimum=1,
                    value=24,
                    scale=1,
                    interactive=True,
                    step=1
                )
                fps = gr.Number(
                    label="FPS",
                    minimum=1,
                    value=1,
                    scale=1,
                    interactive=True,
                    step=1
                )

            with gr.Accordion(
                label="Flow Shift", open=False
            ):
                flow_shift_toggle = gr.Checkbox(
                    label="Enable Flow Shift", value=False
                )
                flow_shift = gr.Number(
                    label="Flow Shift",
                    minimum=1.0,
                    maximum=12.0,
                    value=3.0,
                    interactive=True,
                    step=0.1
                )

            with gr.Row():
                seed_any2video = gr.Number(
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
                with gr.Accordion(
                    label="Image to Video", open=False
                ):
                    img_inp = gr.Image(
                        sources="upload", type="filepath"
                    )
            with gr.Row():
                with gr.Accordion(
                    label="First-Last Frame Video", open=False
                ):
                    with gr.Row():
                        first_frame_inp = gr.Image(
                            sources="upload", type="filepath"
                        )
                    with gr.Row():
                        last_frame_inp = gr.Image(
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
                    label="Status:",
                    visible=False,
                    interactive=False
                )
            with gr.Row():
                video_final = gr.Gallery(
                    label="Generated videos",
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
        any2video,
        inputs=[unet_model, unet_vae, umt5_xxl, clip_vision_h,
                high_noise_model, model_type, taesd_model, phtmkr_model,
                phtmkr_in, phtmkr_nrml, img_inp, first_frame_inp,
                last_frame_inp, upscl, upscl_rep, cnnet, control_img,
                control_strength, pprompt_any2video, nprompt_any2video,
                sampling_any2video, steps_any2video, scheduler,
                width_any2video, height_any2video, batch_count, cfg_any2video,
                frames, fps, flow_shift_toggle, flow_shift, seed_any2video,
                clip_skip, threads, offload_to_cpu, vae_tiling, vae_cpu,
                clip_cpu, cnnet_cpu, canny, rng, predict, output, color,
                flash_attn, diffusion_conv_direct, vae_conv_direct, verbose],
        outputs=[command, progress_slider, progress_textbox, stats,
                 video_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
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
        inputs=[saved_prompts, pprompt_any2video,
                nprompt_any2video],
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
        outputs=[pprompt_any2video, nprompt_any2video]
    )
    random_seed_btn.click(
        random_seed,
        inputs=[],
        outputs=[seed_any2video])

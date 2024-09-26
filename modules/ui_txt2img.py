"""sd.cpp-webui - Text to image UI"""

import os

import gradio as gr

from modules.sdcpp import txt2img
from modules.utility import (
    kill_subprocess, random_seed, sd_tab_switch, flux_tab_switch
)
from modules.config import (
    reload_prompts, save_prompts, delete_prompts, load_prompts,
    emb_dir, lora_dir, taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir
)
from modules.loader import (
    get_models, reload_models
)
from modules.ui import (
    create_model_sel_ui, create_prompts_ui,
    create_cnnet_ui, create_extras_ui, create_settings_ui
)

CURRENT_DIR = os.getcwd()
SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
            "dpm++2mv2", "ipndm", "ipndm_v", "lcm"]
SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits"]
PREDICTION = ["Default", "eps", "v", "flow"]
QUANTS = ["Default", "f32", "f16", "q8_0", "q4_k", "q3_k", "q2_k", "q5_1",
          "q5_0", "q4_1", "q4_0"]
MODELS = ["Stable-Diffusion", "FLUX", "VAE", "clip_l", "t5xxl", "TAESD",
          "Lora", "Embeddings", "Upscaler", "ControlNet"]
RELOAD_SYMBOL = '\U0001f504'
RANDOM_SYMBOL = '\U0001F3B2'
RECYCLE_SYMBOL = '\U0000267C'


with gr.Blocks() as txt2img_block:
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    phtmkr_dir_txt = gr.Textbox(value=phtmkr_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    txt2img_title = gr.Markdown("# Text to Image")

    # Model & VAE Selection
    model_components = create_model_sel_ui()

    # Stable Diffusion Tab Components
    sd_tab = model_components['sd_tab']
    sd_model = model_components['sd_model']
    reload_sd_btn = model_components['reload_sd_btn']
    clear_sd_model = model_components['clear_sd_model']
    sd_vae = model_components['sd_vae']
    reload_vae_btn = model_components['reload_vae_btn']
    clear_vae = model_components['clear_vae']

    # Flux Tab Components
    flux_tab = model_components['flux_tab']
    flux_model = model_components['flux_model']
    reload_flux_btn = model_components['reload_flux_btn']
    clear_flux_model = model_components['clear_flux_model']
    flux_vae = model_components['flux_vae']
    reload_flux_vae_btn = model_components['reload_flux_vae_btn']
    clear_flux_vae = model_components['clear_flux_vae']
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
            value="Default",
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
    pprompt = prompts_components['pprompt']
    nprompt = prompts_components['nprompt']

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            settings_components = create_settings_ui()

            sampling = settings_components['sampling']
            steps = settings_components['steps']
            schedule = settings_components['schedule']
            width = settings_components['width']
            height = settings_components['height']
            batch_count = settings_components['batch_count']
            cfg = settings_components['cfg']

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
            clip_skip = gr.Slider(
                label="CLIP skip",
                minimum=0,
                maximum=12,
                value=0,
                step=1
            )

            # Upscale
            with gr.Accordion(
                label="Upscale", open=False
            ):
                upscl = gr.Dropdown(
                    label="Upscaler",
                    choices=get_models(upscl_dir),
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
            verbose = extras_components['verbose']

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                gen_btn = gr.Button(
                    value="Generate", size="lg"
                )
                kill_btn = gr.Button(
                    value="Stop", size="lg"
                )
            with gr.Row():
                img_final = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    columns=[3],
                    rows=[1],
                    object_fit="contain",
                    height="auto")

    # Generate
    gen_btn.click(
        txt2img,
        inputs=[sd_model, sd_vae, flux_model, flux_vae,
                clip_l, t5xxl, model_type, taesd_model,
                phtmkr_model, phtmkr_in, phtmkr_nrml,
                upscl, upscl_rep, cnnet, control_img,
                control_strength, pprompt, nprompt,
                sampling, steps, schedule, width, height,
                batch_count, cfg, seed, clip_skip, threads,
                vae_tiling, vae_cpu, cnnet_cpu, canny, rng,
                predict, output, color, verbose],
        outputs=[img_final]
    )
    kill_btn.click(
        kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    sd_tab.select(
        sd_tab_switch,
        inputs=[flux_model, flux_vae, clip_l, t5xxl],
        outputs=[sd_model, flux_model, sd_vae, flux_vae, clip_l,
                 t5xxl, pprompt, nprompt]
    )
    flux_tab.select(
        flux_tab_switch,
        inputs=[sd_model, sd_vae, nprompt],
        outputs=[sd_model, flux_model, sd_vae, flux_vae, clip_l,
                 t5xxl, pprompt, nprompt]
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
        inputs=[saved_prompts, pprompt,
                nprompt],
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
        outputs=[pprompt, nprompt]
    )
    random_seed_btn.click(
        random_seed,
        inputs=[],
        outputs=[seed])

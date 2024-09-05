#!/usr/bin/env python3

"""sd.cpp-webui - Main module"""

import os
import argparse

import gradio as gr

from modules.sdcpp import txt2img, img2img, convert
from modules.utility import (
    kill_subprocess, random_seed, sd_tab_switch, flux_tab_switch
)
from modules.gallery import GalleryManager
from modules.config import (
    set_defaults, rst_def, reload_prompts, save_prompts,
    delete_prompts, load_prompts, sd_dir, vae_dir, flux_dir, clip_l_dir,
    t5xxl_dir, emb_dir, lora_dir, taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir,
    txt2img_dir, img2img_dir, def_sd, def_sd_vae, def_flux, def_flux_vae,
    def_clip_l, def_t5xxl, def_sampling, def_steps, def_scheduler, def_width,
    def_height, def_predict
)
from modules.loader import (
    get_models, reload_models, model_choice, model_dir
)
from modules.ui import (
    create_model_sel_ui, create_prompts_ui,
    create_cnnet_ui, create_extras_ui, create_folders_opt_ui,
    create_settings_ui
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


def main():
    """Main"""
    parser = argparse.ArgumentParser(description='Process optional arguments')
    parser.add_argument(
        '--listen',
        action='store_true',
        help='Listen on 0.0.0.0'
    )
    parser.add_argument(
        '--autostart',
        action='store_true',
        help='Automatically launch in a new browser tab'
    )
    args = parser.parse_args()
    sdcpp_launch(args.listen, args.autostart)


def sdcpp_launch(listen=False, autostart=False):
    """Logic for launching sdcpp based on arguments"""
    if listen and autostart:
        print("Launching sdcpp with --listen --autostart")
        sdcpp.launch(server_name="0.0.0.0", inbrowser=True)
    elif listen:
        print("Launching sdcpp with --listen")
        sdcpp.launch(server_name="0.0.0.0")
    elif autostart:
        print("Launching sdcpp with --autostart")
        sdcpp.launch(inbrowser=True)
    else:
        print("Launching sdcpp without any specific options")
        sdcpp.launch()


os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

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
    model_sel = create_model_sel_ui()

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
        with gr.Accordion(
            label="Extra Networks", open=False
        ):
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
                step=0.1
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

    # Generate
    gen_btn.click(
        img2img,
        inputs=[sd_model, sd_vae, flux_model, flux_vae,
                clip_l, t5xxl, model_type, taesd_model,
                phtmkr_model, phtmkr_in, phtmkr_nrml,
                img_inp, upscl, upscl_rep, cnnet,
                control_img, control_strength, pprompt,
                nprompt, sampling, steps, schedule,
                width, height, batch_count,
                strenght, style_ratio, style_ratio_btn,
                cfg, seed, clip_skip, threads, vae_tiling,
                vae_cpu, cnnet_cpu, canny, rng, predict,
                output, color, verbose],
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
        inputs=[saved_prompts, pprompt, nprompt],
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
        outputs=[seed]
    )


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(
        value=0, visible=False
    )
    img2img_ctrl = gr.Textbox(
        value=1, visible=False
    )

    # Title
    gallery_title = gr.Markdown('# Gallery')

    # Gallery Navigation Buttons
    with gr.Row():
        txt2img_btn = gr.Button(value="txt2img")
        img2img_btn = gr.Button(value="img2img")

    with gr.Row():
        pvw_btn = gr.Button(value="Previous")
        nxt_btn = gr.Button(value="Next")

    with gr.Row():
        first_btn = gr.Button(value="First page")
        last_btn = gr.Button(value="Last page")

    with gr.Row():
        page_num_select = gr.Number(
            label="Page:",
            minimum=1,
            value=1,
            interactive=True,
            scale=7
        )
        go_btn = gr.Button(
            value="Go", scale=1
        )

    with gr.Row():
        with gr.Column():
            # Gallery Display
            gallery = gr.Gallery(
                label="txt2img",
                columns=[4],
                rows=[4],
                object_fit="contain",
                height="auto",
                scale=2,
                min_width=500
            )

        with gr.Column():
            # Positive prompts
            pprompt = gr.Textbox(
                label="Positive prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
            )
            # Negative prompts
            nprompt = gr.Textbox(
                label="Negative prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
            )
            # Image Information Display
            img_info_txt = gr.Textbox(
                label="Metadata",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                max_lines=4
            )
            # Delete image Button
            del_img = gr.Button(value="Delete")

    # Interactive bindings
    gallery_manager = GalleryManager(
        txt2img_dir, img2img_dir
    )
    gallery.select(
        gallery_manager.img_info,
        inputs=[],
        outputs=[pprompt, nprompt, img_info_txt]
    )
    txt2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[txt2img_ctrl],
        outputs=[gallery, page_num_select, gallery]
    )
    img2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[img2img_ctrl],
        outputs=[gallery, page_num_select, gallery]
    )
    pvw_btn.click(
        gallery_manager.prev_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    nxt_btn.click(
        gallery_manager.next_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    first_btn.click(
        gallery_manager.reload_gallery,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    last_btn.click(
        gallery_manager.last_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    go_btn.click(
        gallery_manager.goto_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery]
    )
    del_img.click(
        gallery_manager.delete_img,
        inputs=[],
        outputs=[gallery, page_num_select, gallery,
                 pprompt, nprompt, img_info_txt]
    )


with gr.Blocks() as convert_block:
    sd_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    flux_dir_txt = gr.Textbox(value=flux_dir, visible=False)
    clip_l_dir_txt = gr.Textbox(value=clip_l_dir, visible=False)
    t5xxl_dir_txt = gr.Textbox(value=t5xxl_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)
    model_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(
                label="Model Type",
                choices=MODELS,
                interactive=True,
                value="Stable-Diffusion"
            )
            model_type.input(
                model_choice,
                inputs=[model_type],
                outputs=[model_dir_txt]
            )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                model = gr.Dropdown(
                    label="Model",
                    choices=get_models(model_dir),
                    scale=5,
                    interactive=True
                )
                reload_btn = gr.Button(
                    RELOAD_SYMBOL, scale=1
                )
                reload_btn.click(
                    reload_models,
                    inputs=[model_dir_txt],
                    outputs=[model]
                )
            with gr.Row():
                gguf_name = gr.Textbox(
                    label="Output Name (optional, must end with .gguf)",
                    value=""
                )

        with gr.Column():
            with gr.Row():
                quant_type = gr.Dropdown(
                    label="Type",
                    choices=QUANTS,
                    value="f32",
                    interactive=True
                )

            verbose = gr.Checkbox(label="Verbose")

            with gr.Row():
                convert_btn = gr.Button(value="Convert")
                kill_btn = gr.Button(value="Stop")

        # Output
        with gr.Column(scale=1):
            result = gr.Textbox(
                interactive=False,
                value="",
                label="LOG"
            )

    # Interactive Bindings
    convert_btn.click(
        convert,
        inputs=[model, model_dir_txt, quant_type,
                gguf_name, verbose],
        outputs=[result]
    )
    kill_btn.click(
        kill_subprocess,
        inputs=[],
        outputs=[]
    )

    model_dir_txt.change(
        reload_models,
        inputs=[model_dir_txt],
        outputs=[model]
    )


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


sdcpp = gr.TabbedInterface(
    [txt2img_block, img2img_block, gallery_block, convert_block,
     options_block],
    ["txt2img", "img2img", "Gallery", "Checkpoint Converter", "Options"],
    title="sd.cpp-webui",
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    main()

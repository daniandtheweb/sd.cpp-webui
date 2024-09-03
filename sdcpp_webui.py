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
    set_defaults, rst_def, get_prompts, reload_prompts, save_prompts,
    delete_prompts, load_prompts, sd_dir, vae_dir, flux_dir, clip_l_dir,
    t5xxl_dir, emb_dir, lora_dir, taesd_dir, upscl_dir, cnnet_dir, txt2img_dir,
    img2img_dir, def_sd, def_sd_vae, def_flux, def_flux_vae, def_clip_l,
    def_t5xxl, def_sampling, def_steps, def_scheduler, def_width, def_height,
    def_predict
)
from modules.loader import (
    get_models, reload_models, get_hf_models, reload_hf_models
)

CURRENT_DIR = os.getcwd()
SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
            "dpm++2mv2", "ipndm", "ipndm_v", "lcm"]
SCHEDULERS = ["discrete", "karras", "exponential", "ays", "gits"]
PREDICTION = ["Default", "eps", "v", "flow"]
QUANTS = ["Default", "f32", "f16", "q8_0", "q4_k", "q3_k", "q2_k", "q5_1",
          "q5_0", "q4_1", "q4_0"]
RELOAD_SYMBOL = '\U0001f504'
RANDOM_SYMBOL = '\U0001F3B2'
RECYCLE_SYMBOL = '\U0000267C'


def main():
    """Main"""
    parser = argparse.ArgumentParser(description='Process optional arguments')
    parser.add_argument('--listen', action='store_true',
                        help='Listen on 0.0.0.0')
    parser.add_argument('--autostart', action='store_true',
                        help='Automatically launch in a new browser tab')
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

    # Title
    txt2img_title = gr.Markdown("# Text to Image")

    # Model & VAE Selection
    with gr.Row():
        with gr.Tab("Stable Diffusion") as sd_tab:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        sd_model = gr.Dropdown(label="Stable Diffusion Model",
                                               choices=get_models(sd_dir),
                                               scale=7, value=def_sd,
                                               interactive=True)
                    with gr.Row():
                        reload_sd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                        clear_sd_model = gr.ClearButton(sd_model, scale=1)
                with gr.Column():
                    with gr.Row():
                        sd_vae = gr.Dropdown(label="Stable Diffusion VAE",
                                             choices=get_models(vae_dir),
                                             scale=7, value=def_sd_vae,
                                             interactive=True)
                    with gr.Row():
                        reload_vae_btn = gr.Button(value=RELOAD_SYMBOL,
                                                   scale=1)
                        clear_vae = gr.ClearButton(sd_vae, scale=1)
        with gr.Tab("Flux") as flux_tab:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        flux_model = gr.Dropdown(label="Flux Model",
                                                 choices=get_models(flux_dir),
                                                 scale=7, value=def_flux,
                                                 interactive=True)
                    with gr.Row():
                        reload_flux_btn = gr.Button(value=RELOAD_SYMBOL,
                                                    scale=1)
                        clear_flux_model = gr.ClearButton(flux_model, scale=1)
                with gr.Column():
                    with gr.Row():
                        flux_vae = gr.Dropdown(label="Flux VAE",
                                               choices=get_models(vae_dir),
                                               scale=7, value=def_flux_vae,
                                               interactive=True)
                    with gr.Row():
                        reload_vae_btn = gr.Button(value=RELOAD_SYMBOL,
                                                   scale=1)
                        clear_flux_vae = gr.ClearButton(flux_vae, scale=1)
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        clip_l = gr.Dropdown(label="clip_l",
                                             choices=get_models(clip_l_dir),
                                             scale=7, value=def_clip_l,
                                             interactive=True)
                    with gr.Row():
                        reload_clip_l_btn = gr.Button(value=RELOAD_SYMBOL,
                                                      scale=1)
                        clear_clip_l = gr.ClearButton(clip_l, scale=1)
                with gr.Column():
                    with gr.Row():
                        t5xxl = gr.Dropdown(label="t5xxl",
                                            choices=get_models(t5xxl_dir),
                                            scale=7, value=def_t5xxl,
                                            interactive=True)
                    with gr.Row():
                        reload_t5xxl_btn = gr.Button(value=RELOAD_SYMBOL,
                                                     scale=1)
                        clear_t5xxl = gr.ClearButton(t5xxl, scale=1)

    # Model Type Selection
    with gr.Row():
        model_type = gr.Dropdown(label="Quantization", choices=QUANTS,
                                 value="Default")

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                with gr.Column():
                    taesd = gr.Dropdown(label="TAESD",
                                        choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
    with gr.Row():
        with gr.Accordion(label="Saved prompts", open=False):
            with gr.Column():
                saved_prompts = gr.Dropdown(label="Prompts",
                                            choices=get_prompts(),
                                            interactive=True,
                                            allow_custom_value=True)
            with gr.Column():
                with gr.Row():
                    load_prompt_btn = gr.Button(value="Load prompt", size="lg")
                    reload_prompts_btn = gr.Button(value=RELOAD_SYMBOL)
                with gr.Row():
                    save_prompt_btn = gr.Button(value="Save prompt", size="lg")
                    del_prompt_btn = gr.Button(value="Delete prompt",
                                               size="lg")
    with gr.Row():
        pprompt = gr.Textbox(placeholder="Positive prompt",
                             label="Positive Prompt", lines=3,
                             show_copy_button=True)
    with gr.Row():
        nprompt = gr.Textbox(placeholder="Negative prompt",
                             label="Negative Prompt", lines=3,
                             show_copy_button=True)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=SAMPLERS,
                                           value=def_sampling)
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule", choices=SCHEDULERS,
                                       value=def_scheduler)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=2048,
                                      value=def_width, step=64)
                    height = gr.Slider(label="Height", minimum=64,
                                       maximum=2048, value=def_height, step=64)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, value=1, step=1)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            value=7.0, step=0.1)
            with gr.Row():
                seed = gr.Number(label="Seed", minimum=-1, maximum=10**16,
                                 value=-1, scale=5)
                random_seed_btn = gr.Button(value=RANDOM_SYMBOL, scale=1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=1)

            # Upscale
            with gr.Accordion(label="Upscale", open=False):
                upscl = gr.Dropdown(label="Upscaler",
                                    choices=get_models(upscl_dir))
                reload_upscl_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_upscl = gr.ClearButton(upscl)
                upscl_rep = gr.Slider(label="Upscaler repeats", minimum=1,
                                      maximum=5, value=1, step=0.1)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_cnnet = gr.ClearButton(cnnet)
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
                cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")

            # Extra Settings
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                vae_tiling = gr.Checkbox(label="VAE Tiling")
                vae_cpu = gr.Checkbox(label="VAE on CPU")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                predict = gr.Dropdown(label="Prediction",
                                      choices=PREDICTION,
                                      value=def_predict)
                output = gr.Textbox(label="Output Name",
                                    placeholder="Optional")
                color = gr.Checkbox(label="Color", value=True)
                verbose = gr.Checkbox(label="Verbose")

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                gen_btn = gr.Button(value="Generate", size="lg")
                kill_btn = gr.Button(value="Stop", size="lg")
            with gr.Row():
                img_final = gr.Gallery(label="Generated images",
                                       show_label=False, columns=[3], rows=[1],
                                       object_fit="contain", height="auto")

    # Generate
    gen_btn.click(txt2img, inputs=[sd_model, sd_vae, flux_model, flux_vae,
                                   clip_l, t5xxl, model_type, taesd, upscl,
                                   upscl_rep, cnnet, control_img,
                                   control_strength, pprompt, nprompt,
                                   sampling, steps, schedule, width, height,
                                   batch_count, cfg, seed, clip_skip, threads,
                                   vae_tiling, vae_cpu, cnnet_cpu, rng,
                                   predict, output, color, verbose],
                  outputs=[img_final])
    kill_btn.click(kill_subprocess, inputs=[], outputs=[])

    # Interactive Bindings
    sd_tab.select(sd_tab_switch, inputs=[flux_model, flux_vae, clip_l, t5xxl],
                  outputs=[sd_model, flux_model, sd_vae, flux_vae, clip_l,
                           t5xxl, pprompt, nprompt])
    flux_tab.select(flux_tab_switch, inputs=[sd_model, sd_vae, nprompt],
                    outputs=[sd_model, flux_model, sd_vae, flux_vae, clip_l,
                             t5xxl, pprompt, nprompt])
    reload_sd_btn.click(reload_models, inputs=[sd_dir_txt],
                        outputs=[sd_model])
    reload_flux_btn.click(reload_models, inputs=[flux_dir_txt],
                          outputs=[flux_model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[sd_vae])
    reload_clip_l_btn.click(reload_models, inputs=[clip_l_dir_txt],
                            outputs=[clip_l])
    reload_t5xxl_btn.click(reload_models, inputs=[t5xxl_dir_txt],
                           outputs=[t5xxl])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_upscl_btn.click(reload_models, inputs=[upscl_dir_txt],
                           outputs=[upscl])
    reload_cnnet_btn.click(reload_models, inputs=[cnnet_dir_txt],
                           outputs=[cnnet])
    save_prompt_btn.click(save_prompts, inputs=[saved_prompts, pprompt,
                                                nprompt], outputs=[])
    del_prompt_btn.click(delete_prompts, inputs=[saved_prompts],
                         outputs=[])
    reload_prompts_btn.click(reload_prompts, inputs=[],
                             outputs=[saved_prompts])
    load_prompt_btn.click(load_prompts, inputs=[saved_prompts],
                          outputs=[pprompt, nprompt])
    random_seed_btn.click(random_seed, [], [seed])


with gr.Blocks()as img2img_block:
    # Directory Textboxes
    sd_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    img2img_title = gr.Markdown("# Image to Image")

    # Model & VAE Selection
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model = gr.Dropdown(label="Model",
                                    choices=get_models(sd_dir), scale=7,
                                    value=def_sd)
            with gr.Row():
                reload_sd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
        with gr.Column():
            with gr.Row():
                sd_vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir),
                                     scale=7, value=def_sd_vae)
            with gr.Row():
                reload_vae_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                clear_vae = gr.ClearButton(sd_vae, scale=1)

    # Model Type Selection
    with gr.Row():
        model_type = gr.Dropdown(label="Quantization", choices=QUANTS,
                                 value="Default")

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                with gr.Column():
                    taesd = gr.Dropdown(label="TAESD",
                                        choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
    with gr.Row():
        with gr.Accordion(label="Saved prompts", open=False):
            saved_prompts = gr.Dropdown(label="Prompts",
                                        choices=get_prompts(),
                                        interactive=True,
                                        allow_custom_value=True)
            with gr.Column():
                with gr.Row():
                    load_prompt_btn = gr.Button(value="Load prompt", size="lg")
                    reload_prompts_btn = gr.Button(value=RELOAD_SYMBOL)
                with gr.Row():
                    save_prompt_btn = gr.Button(value="Save prompt", size="lg")
                    del_prompt_btn = gr.Button(value="Delete prompt",
                                               size="lg")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                pprompt = gr.Textbox(placeholder="Positive prompt",
                                     label="Positive Prompt", lines=3,
                                     show_copy_button=True)
            with gr.Row():
                nprompt = gr.Textbox(placeholder="Negative prompt",
                                     label="Negative Prompt", lines=3,
                                     show_copy_button=True)
        with gr.Column():
            img_inp = gr.Image(sources="upload", type="filepath")

    # Settings
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=SAMPLERS,
                                           value=def_sampling)
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule",
                                       choices=SCHEDULERS,
                                       value=def_scheduler)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=2048,
                                      value=def_width, step=8)
                    height = gr.Slider(label="Height", minimum=64,
                                       maximum=2048, value=def_height, step=8)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, step=1, value=1)
            strenght = gr.Slider(label="Noise strenght", minimum=0, maximum=1,
                                 step=0.01, value=0.75)
            style_ratio_btn = gr.Checkbox(label="Enable style-ratio")
            style_ratio = gr.Slider(label="Style ratio", minimum=0,
                                    maximum=100, step=1, value=20)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            step=0.1, value=7.0)
            with gr.Row():
                seed = gr.Number(label="Seed", minimum=-1, maximum=10**16,
                                 value=-1, scale=5)
                random_seed_btn = gr.Button(value=RANDOM_SYMBOL, scale=1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)

            # Upscale
            with gr.Accordion(label="Upscale", open=False):
                upscl = gr.Dropdown(label="Upscaler",
                                    choices=get_models(upscl_dir))
                reload_upscl_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_upscl = gr.ClearButton(upscl)
                upscl_rep = gr.Slider(label="Upscaler repeats", minimum=1,
                                      maximum=5, value=1, step=0.1)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_connet = gr.ClearButton(cnnet)
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
                cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")
                canny = gr.Checkbox(label="Canny (edge detection)")

            # Extra Settings
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                vae_tiling = gr.Checkbox(label="VAE Tiling")
                vae_cpu = gr.Checkbox(label="VAE on CPU")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                predict = gr.Dropdown(label="Prediction",
                                      choices=PREDICTION,
                                      value=def_predict)
                output = gr.Textbox(label="Output Name (optional)", value="")
                color = gr.Checkbox(label="Color", value=True)
                verbose = gr.Checkbox(label="Verbose")
        with gr.Column(scale=1):
            with gr.Row():
                gen_btn = gr.Button(value="Generate")
                kill_btn = gr.Button(value="Stop")
            with gr.Row():
                img_final = gr.Gallery(label="Generated images",
                                       show_label=False, columns=[3], rows=[1],
                                       object_fit="contain", height="auto")

    # Generate
    gen_btn.click(img2img, inputs=[model, sd_vae, model_type, taesd, img_inp,
                                   upscl, upscl_rep, cnnet, control_img,
                                   control_strength, pprompt,
                                   nprompt, sampling, steps, schedule,
                                   width, height, batch_count,
                                   strenght, style_ratio, style_ratio_btn,
                                   cfg, seed, clip_skip, threads, vae_tiling,
                                   vae_cpu, cnnet_cpu, canny, rng, predict,
                                   output, color, verbose],
                  outputs=[img_final])
    kill_btn.click(kill_subprocess, inputs=[], outputs=[])

    # Interactive Bindings
    reload_sd_btn.click(reload_models, inputs=[sd_dir_txt],
                        outputs=[model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[sd_vae])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_upscl_btn.click(reload_models, inputs=[upscl_dir_txt],
                           outputs=[upscl])
    reload_cnnet_btn.click(reload_models, inputs=[cnnet_dir_txt],
                           outputs=[cnnet])
    random_seed_btn.click(random_seed, [], [seed])


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(value=0, visible=False)
    img2img_ctrl = gr.Textbox(value=1, visible=False)

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
        page_num_select = gr.Number(label="Page:", minimum=1, value=1,
                                    interactive=True, scale=7)
        go_btn = gr.Button(value="Go", scale=1)

    with gr.Row():
        with gr.Column():
            # Gallery Display
            gallery = gr.Gallery(label="txt2img", columns=[4], rows=[4],
                                 object_fit="contain", height="auto",
                                 scale=2, min_width=500)

        with gr.Column():
            # Positive prompts
            pprompt = gr.Textbox(label="Positive prompt:", value="",
                                 interactive=False, scale=1,
                                 min_width=300, show_copy_button=True,
                                 max_lines=4)
            nprompt = gr.Textbox(label="Negative prompt:", value="",
                                 interactive=False, scale=1,
                                 min_width=300, show_copy_button=True,
                                 max_lines=4)
            # Negative prompts
            # Image Information Display
            img_info_txt = gr.Textbox(label="Metadata", value="",
                                      interactive=False, scale=1,
                                      min_width=300, max_lines=4)
            # Delete image Button
            del_img = gr.Button(value="Delete")

    # Interactive bindings
    gallery_manager = GalleryManager(txt2img_dir, img2img_dir)
    gallery.select(gallery_manager.img_info, inputs=[],
                   outputs=[pprompt, nprompt, img_info_txt])
    txt2img_btn.click(gallery_manager.reload_gallery, inputs=[txt2img_ctrl],
                      outputs=[gallery, page_num_select, gallery])
    img2img_btn.click(gallery_manager.reload_gallery, inputs=[img2img_ctrl],
                      outputs=[gallery, page_num_select, gallery])
    pvw_btn.click(gallery_manager.prev_page, inputs=[],
                  outputs=[gallery, page_num_select, gallery])
    nxt_btn.click(gallery_manager.next_page, inputs=[],
                  outputs=[gallery, page_num_select, gallery])
    first_btn.click(gallery_manager.reload_gallery, inputs=[],
                    outputs=[gallery, page_num_select, gallery])
    last_btn.click(gallery_manager.last_page, inputs=[],
                   outputs=[gallery, page_num_select, gallery])
    go_btn.click(gallery_manager.goto_gallery, inputs=[page_num_select],
                 outputs=[gallery, page_num_select, gallery])
    del_img.click(gallery_manager.delete_img, inputs=[],
                  outputs=[gallery, page_num_select, gallery,
                           pprompt, nprompt, img_info_txt])


with gr.Blocks() as convert_block:
    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        # Input
        with gr.Column(scale=1):
            with gr.Row():
                orig_model = gr.Dropdown(label="Original Model",
                                         choices=get_hf_models(), scale=5)
                reload_btn = gr.Button(RELOAD_SYMBOL, scale=1)
                reload_btn.click(reload_hf_models, inputs=[],
                                 outputs=[orig_model])

            quant_type = gr.Dropdown(label="Type",
                                     choices=QUANTS,
                                     value="f32")

            verbose = gr.Checkbox(label="Verbose")

            gguf_name = gr.Textbox(label="Output Name (optional, must end "
                                   "with .gguf)", value="")

            with gr.Row():
                convert_btn = gr.Button(value="Convert")
                kill_btn = gr.Button(value="Stop")

        # Output
        with gr.Column(scale=1):
            result = gr.Textbox(interactive=False, value="")

    # Interactive Bindings
    convert_btn.click(convert, inputs=[orig_model, quant_type, gguf_name,
                                       verbose], outputs=[result])
    kill_btn.click(kill_subprocess, [], [])


with gr.Blocks() as options_block:
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                sd_model = gr.Dropdown(label="Stable Diffusion Model",
                                       choices=get_models(sd_dir),
                                       scale=7, value=def_sd,
                                       interactive=True)
            with gr.Row():
                reload_sd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                clear_sd_model = gr.ClearButton(sd_model, scale=1)
        with gr.Column():
            with gr.Row():
                sd_vae = gr.Dropdown(label="Stable Diffusion VAE",
                                     choices=get_models(vae_dir), scale=7,
                                     value=def_sd_vae, interactive=True)
            with gr.Row():
                reload_vae_btn = gr.Button(value=RELOAD_SYMBOL,
                                           scale=1)
                clear_vae = gr.ClearButton(sd_vae, scale=1)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                flux_model = gr.Dropdown(label="Flux Model",
                                         choices=get_models(flux_dir),
                                         scale=7, value=def_flux,
                                         interactive=True)
            with gr.Row():
                reload_flux_btn = gr.Button(value=RELOAD_SYMBOL,
                                            scale=1)
                clear_flux_model = gr.ClearButton(flux_model, scale=1)
        with gr.Column():
            with gr.Row():
                flux_vae = gr.Dropdown(label="Flux VAE",
                                       choices=get_models(vae_dir),
                                       scale=7, value=def_flux_vae,
                                       interactive=True)
            with gr.Row():
                reload_vae_btn = gr.Button(value=RELOAD_SYMBOL,
                                           scale=1)
                clear_flux_vae = gr.ClearButton(flux_vae, scale=1)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                clip_l = gr.Dropdown(label="clip_l",
                                     choices=get_models(clip_l_dir),
                                     scale=7, value=def_clip_l,
                                     interactive=True)
            with gr.Row():
                reload_clip_l_btn = gr.Button(value=RELOAD_SYMBOL,
                                              scale=1)
                clear_clip_l = gr.ClearButton(clip_l, scale=1)
        with gr.Column():
            with gr.Row():
                t5xxl = gr.Dropdown(label="t5xxl",
                                    choices=get_models(t5xxl_dir),
                                    scale=7, value=def_t5xxl,
                                    interactive=True)
            with gr.Row():
                reload_t5xxl_btn = gr.Button(value=RELOAD_SYMBOL,
                                             scale=1)
                clear_t5xxl = gr.ClearButton(t5xxl, scale=1)

    with gr.Row():
        with gr.Column():
            # Sampling Method Dropdown
            sampling = gr.Dropdown(label="Sampling method",
                                   choices=SAMPLERS, value=def_sampling,
                                   interactive=True)
        with gr.Column():
            # Steps Slider
            steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                              value=def_steps, step=1)

    with gr.Row():
        # Schedule Dropdown
        schedule = gr.Dropdown(label="Schedule",
                               choices=SCHEDULERS,
                               value="discrete", interactive=True)

    with gr.Column():
        # Size Sliders
        width = gr.Slider(label="Width", minimum=64, maximum=2048,
                          value=def_width, step=8)
        height = gr.Slider(label="Height", minimum=64, maximum=2048,
                           value=def_height, step=8)

    with gr.Row():
        # Prediction mode
        predict = gr.Dropdown(label="Prediction", choices=PREDICTION,
                              value=def_predict, interactive=True)

    with gr.Row():
        # Folders Accordion
        with gr.Accordion(label="Folders", open=False):
            sd_dir_txt = gr.Textbox(label="Stable Diffusion folder",
                                    value=sd_dir, interactive=True)
            flux_dir_txt = gr.Textbox(label="Flux folder", value=flux_dir,
                                      interactive=True)
            vae_dir_txt = gr.Textbox(label="VAE folder", value=vae_dir,
                                     interactive=True)
            clip_l_dir_txt = gr.Textbox(label="clip_l folder",
                                        value=clip_l_dir, interactive=True)
            t5xxl_dir_txt = gr.Textbox(label="t5xxl folder", value=t5xxl_dir,
                                       interactive=True)
            emb_dir_txt = gr.Textbox(label="Embeddings folder", value=emb_dir,
                                     interactive=True)
            lora_dir_txt = gr.Textbox(label="Lora folder", value=lora_dir,
                                      interactive=True)
            taesd_dir_txt = gr.Textbox(label="TAESD folder", value=taesd_dir,
                                       interactive=True)
            upscl_dir_txt = gr.Textbox(label="Upscaler folder",
                                       value=upscl_dir, interactive=True)
            cnnet_dir_txt = gr.Textbox(label="ControlNet folder",
                                       value=cnnet_dir, interactive=True)
            txt2img_dir_txt = gr.Textbox(label="txt2img outputs folder",
                                         value=txt2img_dir, interactive=True)
            img2img_dir_txt = gr.Textbox(label="img2img outputs folder",
                                         value=img2img_dir, interactive=True)

    # Set Defaults and Restore Defaults Buttons
    with gr.Row():
        set_btn = gr.Button(value="Set Defaults")
        set_btn.click(set_defaults, [sd_model, sd_vae, flux_model, flux_vae,
                                     clip_l, t5xxl, sampling, steps, schedule,
                                     width, height, predict,
                                     sd_dir_txt, flux_dir_txt, vae_dir_txt,
                                     clip_l_dir_txt, t5xxl_dir_txt,
                                     emb_dir_txt, lora_dir_txt,
                                     taesd_dir_txt, upscl_dir_txt,
                                     cnnet_dir_txt, txt2img_dir_txt,
                                     img2img_dir_txt], [])
        restore_btn = gr.Button(value="Restore Defaults")
        restore_btn.click(rst_def, [], [])


sdcpp = gr.TabbedInterface(
    [txt2img_block, img2img_block, gallery_block, convert_block,
     options_block],
    ["txt2img", "img2img", "Gallery", "Checkpoint Converter", "Options"],
    title="sd.cpp-webui",
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    main()

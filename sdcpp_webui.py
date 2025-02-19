#!/usr/bin/env python3

"""sd.cpp-webui - Main module"""

import os
import argparse

import gradio as gr

from modules.ui_txt2img import txt2img_block, pprompt_txt2img, nprompt_txt2img, width_txt2img, height_txt2img, steps_txt2img, sampling_txt2img
from modules.ui_img2img import img2img_block, pprompt_img2img, nprompt_img2img, width_img2img, height_img2img, steps_img2img, sampling_img2img
from modules.ui_gallery import gallery_block, cpy_2_txt2img_btn, cpy_2_img2img_btn, pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info
from modules.ui_convert import convert_block
from modules.ui_options import options_block


os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'


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
    parser.add_argument(
        '--darkmode',
        action='store_true',
        help='Enable dark mode for the web interface'
    )
    args = parser.parse_args()
    sdcpp_launch(args.listen, args.autostart, args.darkmode)


def cpy_2_txt2img(pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info):
    return gr.Tabs(selected="txt2img"), pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info


def cpy_2_img2img(pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info):
    return gr.Tabs(selected="img2img"), pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info


def sdcpp_launch(
        listen=False, autostart=False, darkmode=False
):
    """Logic for launching sdcpp based on arguments"""
    launch_args = {}

    if listen:
        launch_args["server_name"] = "0.0.0.0"
    if autostart:
        launch_args["inbrowser"] = True

    # this js forces the url to redirect to the darkmode link
    dark_js = """
    function refresh() {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """ if darkmode else None


    with gr.Blocks(css="footer {visibility: hidden}", title="sd.cpp-webui", theme="default", js=dark_js) as sdcpp:
        with gr.Tabs() as tabs:
            with gr.TabItem("txt2img", id="txt2img"):
                txt2img_block.render()
            with gr.TabItem("img2img", id="img2img"):
                img2img_block.render()
            with gr.TabItem("Gallery", id="gallery"):
                gallery_block.render()
            with gr.TabItem("Checkpoint Converter", id="convert"):
                convert_block.render()
            with gr.TabItem("Options", id="options"):
                options_block.render()

        # Set up the button click event
        cpy_2_txt2img_btn.click(
            cpy_2_txt2img,
            inputs=[pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info],
            outputs=[tabs, pprompt_txt2img, nprompt_txt2img, width_txt2img, height_txt2img, steps_txt2img, sampling_txt2img]
        )
        cpy_2_img2img_btn.click(
            cpy_2_img2img,
            inputs=[pprompt_info, nprompt_info, width_info, height_info, steps_info, sampler_info],
            outputs=[tabs, pprompt_img2img, nprompt_img2img, width_img2img, height_img2img, steps_img2img, sampling_img2img]
        )

    # Pass the arguments to sdcpp.launch with argument unpacking
    sdcpp.launch(**launch_args)


if __name__ == "__main__":
    main()

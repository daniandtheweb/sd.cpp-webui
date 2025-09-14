#!/usr/bin/env python3

"""sd.cpp-webui - Main module"""

import os
import argparse

import gradio as gr

from modules.ui_txt2img import (
    txt2img_block, pprompt_txt2img, nprompt_txt2img, width_txt2img,
    height_txt2img, steps_txt2img, sampling_txt2img, scheduler_txt2img,
    cfg_txt2img, seed_txt2img
)
from modules.ui_img2img import (
    img2img_block, pprompt_img2img, nprompt_img2img, width_img2img,
    height_img2img, steps_img2img, sampling_img2img, scheduler_img2img,
    cfg_img2img, seed_img2img, img_inp
)
from modules.ui_any2video import (
    any2video_block, pprompt_any2video, nprompt_any2video, width_any2video,
    height_any2video, steps_any2video, sampling_any2video, scheduler_any2video,
    cfg_any2video, seed_any2video
)
from modules.ui_gallery import (
    gallery_block, cpy_2_txt2img_btn, cpy_2_img2img_btn, cpy_2_any2video_btn,
    pprompt_info, nprompt_info, width_info, height_info, steps_info,
    sampler_info, scheduler_info, cfg_info, seed_info, path_info
)
from modules.ui_convert import convert_block
from modules.ui_options import options_block
from modules.config import ConfigManager


os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

config = ConfigManager()


def create_copy_fn(tab_id: str) -> callable:
    """
    Creates a function that that switches to a specific tab
    and passes through its arguments.

    Args:
        tab_id (str): The ID of the selected tab.

    Returns:
        A function suitable for a Gradio .click() event.
    """
    def copy_fn(*args):
        # The first return value switches the tab,
        # the rest are the passed-through arguments.
        return [gr.Tabs(selected=tab_id), *args]
    return copy_fn


def sdcpp_launch(
    listen: bool = False, autostart: bool = False, darkmode: bool = False
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

    with gr.Blocks(
        css="footer {visibility: hidden}", title="sd.cpp-webui",
        theme="default", js=dark_js
    ) as sdcpp:
        gr.Markdown("# <center>sd.cpp-webui</center>")
        with gr.Tabs() as tabs:
            with gr.TabItem("txt2img", id="txt2img"):
                txt2img_block.render()
            with gr.TabItem("img2img", id="img2img"):
                img2img_block.render()
            with gr.TabItem("any2video", id="any2video"):
                any2video_block.render()
            with gr.TabItem("Gallery", id="gallery"):
                gallery_block.render()
            with gr.TabItem("Checkpoint Converter", id="convert"):
                convert_block.render()
            with gr.TabItem("Options", id="options"):
                options_block.render()

        common_inputs = [
            pprompt_info, nprompt_info, width_info, height_info,
            steps_info, sampler_info, scheduler_info, cfg_info, seed_info
        ]

        # Copy data from gallery image to txt2img.
        cpy_2_txt2img_btn.click(
            create_copy_fn("txt2img"),
            inputs=common_inputs,
            outputs=[
                tabs, pprompt_txt2img, nprompt_txt2img, width_txt2img,
                height_txt2img, steps_txt2img, sampling_txt2img,
                scheduler_txt2img, cfg_txt2img, seed_txt2img
            ]
        )
        # Copy data from gallery image to img2img.
        cpy_2_img2img_btn.click(
            create_copy_fn("img2img"),
            inputs=common_inputs + [path_info],
            outputs=[
                tabs, pprompt_img2img, nprompt_img2img, width_img2img,
                height_img2img, steps_img2img, sampling_img2img,
                scheduler_img2img, cfg_img2img, seed_img2img, img_inp
            ]
        )
        # Copy data from gallery image to any2video.
        cpy_2_any2video_btn.click(
            create_copy_fn("any2video"),
            inputs=common_inputs + [path_info],
            outputs=[
                tabs, pprompt_any2video, nprompt_any2video,
                width_any2video, height_any2video, steps_any2video,
                sampling_any2video, scheduler_any2video, cfg_any2video,
                seed_any2video, img_inp]
        )

    # Pass the arguments to sdcpp.launch with argument unpacking
    sdcpp.launch(**launch_args)


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


if __name__ == "__main__":
    main()

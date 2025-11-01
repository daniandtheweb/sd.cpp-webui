#!/usr/bin/env python3

"""sd.cpp-webui - Main module"""

import os
import argparse

import gradio as gr

from modules.txt2img_ui import (
    txt2img_block, txt2img_params
)
from modules.img2img_ui import (
    img2img_block, img2img_params, img_inp_img2img
)
from modules.imgedit_ui import (
    imgedit_block, width_imgedit, height_imgedit, ref_img_imgedit
)
from modules.any2video_ui import (
    any2video_block, any2video_params
)
from modules.upscale_ui import upscale_block, img_inp_upscale
from modules.gallery_ui import (
    gallery_block, cpy_2_txt2img_btn, cpy_2_img2img_btn, cpy_2_imgedit_btn,
    cpy_2_any2video_btn, cpy_2_upscale_btn, info_params, path_info
)
from modules.convert_ui import convert_block
from modules.options_ui import options_block
from modules.config import ConfigManager
from modules.ui.constants import FIELDS, SAMPLERS, SCHEDULERS


os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

config = ConfigManager()


def create_copy_fn(tab_id: str, fields: list = None) -> callable:
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
        values = args
        if fields:
            values = []
            for name, value in zip(fields, args):
                if name == 'sampling' and value not in SAMPLERS:
                    value = config.get('def_sampling')
                elif name == 'scheduler' and value not in SCHEDULERS:
                    value = config.get('def_scheduler')
                values.append(value)
        return [gr.Tabs(selected=tab_id), *values]
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
            with gr.TabItem("imgedit", id="imgedit"):
                imgedit_block.render()
            with gr.TabItem("any2video", id="any2video"):
                any2video_block.render()
            with gr.TabItem("Gallery", id="gallery"):
                gallery_block.render()
            with gr.TabItem("Upscaler", id="upscale"):
                upscale_block.render()
            with gr.TabItem("Checkpoint Converter", id="convert"):
                convert_block.render()
            with gr.TabItem("Options", id="options"):
                options_block.render()

        common_inputs = [info_params[f] for f in FIELDS]

        # Copy data from gallery image to txt2img.
        cpy_2_txt2img_btn.click(
            create_copy_fn("txt2img", FIELDS),
            inputs=common_inputs,
            outputs=[tabs] + [txt2img_params[f] for f in FIELDS]
        )
        # Copy data from gallery image to img2img.
        cpy_2_img2img_btn.click(
            create_copy_fn("img2img", FIELDS + ['input_image']),
            inputs=common_inputs + [path_info],
            outputs=[tabs] + [img2img_params[f] for f in FIELDS] + [img_inp_img2img]
        )
        # Copy data from gallery image to imgedit
        cpy_2_imgedit_btn.click(
            create_copy_fn("imgedit"),
            inputs=[info_params['width'], info_params['height'], path_info],
            outputs=[tabs, width_imgedit, height_imgedit, ref_img_imgedit]
        )
        # Copy data from gallery image to any2video.
        cpy_2_any2video_btn.click(
            create_copy_fn("any2video", FIELDS),
            inputs=common_inputs + [path_info],
            outputs=[tabs] + [any2video_params[f] for f in FIELDS]
        )
        cpy_2_upscale_btn.click(
            create_copy_fn("upscale"),
            inputs=[path_info],
            outputs=[tabs, img_inp_upscale]
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

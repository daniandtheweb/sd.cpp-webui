#!/usr/bin/env python3

"""sd.cpp-webui - Main module"""

import os
import argparse

import gradio as gr

from modules.ui_txt2img import txt2img_block
from modules.ui_img2img import img2img_block
from modules.ui_gallery import gallery_block
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

    sdcpp = gr.TabbedInterface(
        [txt2img_block, img2img_block, gallery_block, convert_block,
         options_block],
        ["txt2img", "img2img", "Gallery", "Checkpoint Converter", "Options"],
        title="sd.cpp-webui",
        theme="default",
        js=dark_js,
        css="footer {visibility: hidden}",
    )

    # Pass the arguments to sdcpp.launch with argument unpacking
    sdcpp.launch(**launch_args)


if __name__ == "__main__":
    main()

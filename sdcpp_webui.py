#!/usr/bin/env python3
"""sd.cpp-webui - Main module"""

import os
import sys
import json
import argparse

import gradio as gr

from modules.interfaces.cli.txt2img_tab import (
    txt2img_block, txt2img_params
)
from modules.interfaces.server.txt2img_tab import (
    txt2img_server_block
)
from modules.interfaces.cli.img2img_tab import (
    img2img_block, img2img_params, img_inp_img2img
)
from modules.interfaces.server.img2img_tab import (
    img2img_server_block
)
from modules.interfaces.cli.imgedit_tab import (
    imgedit_block, width_imgedit, height_imgedit, ref_img_imgedit
)
from modules.interfaces.server.imgedit_tab import (
    imgedit_server_block
)
from modules.interfaces.cli.any2video_tab import (
    any2video_block, any2video_params
)
from modules.interfaces.server.any2video_tab import (
    any2video_server_block
)
from modules.interfaces.cli.upscale_tab import (
    upscale_block, img_inp_upscale
)
from modules.interfaces.server.upscale_tab import (
    upscale_server_block
)
from modules.interfaces.common.gallery_tab import (
    gallery_block, cpy_2_txt2img_btn, cpy_2_img2img_btn, cpy_2_imgedit_btn,
    cpy_2_any2video_btn, cpy_2_upscale_btn, info_params, path_info,
    gallery, gallery_manager, def_page, txt2img_ctrl, page_num_select
)
from modules.interfaces.cli.convert_tab import convert_block
from modules.interfaces.common.options_tab import (
    options_block, restart_btn
)
from modules.config import ConfigManager
from modules.ui.constants import FIELDS, SAMPLERS, SCHEDULERS


os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
config = ConfigManager()

theme = config.get('def_theme')


def create_copy_fn(tab_id: str, fields: list = None) -> callable:
    """
    Creates a function that switches to a specific tab
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


def lazy_load_gallery(is_loaded, page, ctrl):
    if is_loaded:
        # If already loaded, return existing values (do nothing)
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), True

    results = gallery_manager.reload_gallery(page, ctrl)

    return *results, True


def load_credentials(filepath: str = "credentials.json"):
    """
    Loads usernames and passwords from a JSON file.
    Expected format: {"username1": "password1", "username2": "password2"}
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                data = json.load(file)

            # Gradio expects auth to be a list of tuples: [("user", "pass"), ...]
            return list(data.items())
        else:
            print(f"Credentials file '{filepath}' not found. Skipping password protection.")
            return None
    except Exception as e:
        print(f"Error reading credentials: {e}. Skipping password protection.")
        return None


def restart_server():
    """
    Restarts the sdcpp-webui.
    """
    print("\nRestarting server...")
    os.environ['SDCPP_IS_RESTART'] = 'true'
    python = sys.executable
    new_args = [arg for arg in sys.argv if arg != '--autostart']
    os.execv(python, [python] + new_args)


def sdcpp_launch(
    server: bool = False, listen: bool = False,
    autostart: bool = False, darkmode: bool = False,
    credentials: bool = False, insecure_dir: bool = False
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

    if credentials:
        auth_data = load_credentials()
        if auth_data:
            print(f"Secure mode enabled with {len(auth_data)} users.")
            launch_args["auth"] = auth_data
        else:
            print("Secure mode requested but failed to load credentials. Launching without auth.")

    if insecure_dir:
        allowed_paths = []

        base_path = os.path.abspath(os.getcwd())

        dirs = [
            val for key, val in config.data.items()
            if key.endswith('_dir') and isinstance(val, str) and val
        ]

        for path in dirs:
            # Expand user tildes
            expanded_path = os.path.expanduser(path)

            abs_path = os.path.abspath(expanded_path)

            # Check if it's a symlink (STRIP TRAILING SLASHES)
            # os.path.islink() returns False if the path ends with a separator
            is_link = os.path.islink(abs_path.rstrip(os.sep))

            # Check if it is physically outside the base path
            real_path = os.path.realpath(abs_path)
            is_external = not real_path.startswith(base_path)

            if is_link or is_external:
                if is_link:
                    allowed_paths.append(real_path)
                if abs_path != real_path:
                    allowed_paths.append(abs_path)
                elif not is_link:
                    allowed_paths.append(abs_path)

        # Remove duplicates
        allowed_paths = list(set(allowed_paths))

        if allowed_paths:
            print("Allowing external/linked directories:")
            for path in allowed_paths:
                print(f" - {path}")
            print()

        launch_args["allowed_paths"] = allowed_paths

    with gr.Blocks(
        css="footer {visibility: hidden}", title="sd.cpp-webui",
        theme=theme, js=dark_js
    ) as sdcpp:

        gallery_loaded_state = gr.State(value=False)
        common_inputs = [info_params[f] for f in FIELDS]

        if server:
            gr.Markdown("# <center>sd.cpp-webui - server</center>")
        else:
            gr.Markdown("# <center>sd.cpp-webui - cli</center>")
        with gr.Tabs() as tabs:
            with gr.TabItem("txt2img", id="txt2img"):
                if server:
                    txt2img_server_block.render()
                else:
                    txt2img_block.render()
            with gr.TabItem("img2img", id="img2img"):
                if server:
                    img2img_server_block.render()
                else:
                    img2img_block.render()
            with gr.TabItem("imgedit", id="imgedit"):
                if server:
                    imgedit_server_block.render()
                else:
                    imgedit_block.render()
            with gr.TabItem("any2video", id="any2video"):
                if server:
                    any2video_server_block.render()
                else:
                    any2video_block.render()
            with gr.TabItem("Gallery", id="gallery") as gallery_tab:
                gallery_block.render()
            with gr.TabItem("Upscaler", id="upscale"):
                if server:
                    upscale_server_block.render()
                else:
                    upscale_block.render()
            if not server:
                with gr.TabItem("Checkpoint Converter", id="convert"):
                    convert_block.render()
            with gr.TabItem("Options", id="options"):
                options_block.render()

        gallery_tab.select(
            fn=lazy_load_gallery,
            inputs=[gallery_loaded_state, def_page, txt2img_ctrl],
            outputs=[
                gallery, page_num_select, gallery,
                gallery, gallery_loaded_state
            ]
        )
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

        restart_btn.click(
            fn=restart_server,
            inputs=[],
            outputs=[]
        )

    # Pass the arguments to sdcpp.launch with argument unpacking
    sdcpp.launch(**launch_args)


def main():
    """Main"""
    if os.environ.get('SDCPP_IS_RESTART') == 'true':
        print("\n" + "="*60)
        print(" SERVER RESTARTED SUCCESSFULLY")
        print(" Please refresh your web browser to apply the changes.")
        print("="*60 + "\n")

        del os.environ['SDCPP_IS_RESTART']

    parser = argparse.ArgumentParser(description='Process optional arguments')
    parser.add_argument(
        '--server',
        action='store_true',
        help='Run stable-diffusion.cpp\'s server mode'
    )
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
    parser.add_argument(
        '--credentials',
        action='store_true',
        help='Enable password protection using credentials.json'
    )
    parser.add_argument(
        '--allow-insecure-dir',
        action='store_true',
        help='Allows the usage of external or linked directories based on config.json'
    )
    args = parser.parse_args()

    sdcpp_launch(
        args.server, args.listen, args.autostart, args.darkmode,
        args.credentials, args.allow_insecure_dir
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

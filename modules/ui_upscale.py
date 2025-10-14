"""sd.cpp-webui - Upscale UI"""

import gradio as gr
from PIL import Image

from modules.sdcpp import upscale
from modules.utility import (
    subprocess_manager, switch_sizes
)
from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.ui import (
    RELOAD_SYMBOL, SWITCH_V_SYMBOL
)


with gr.Blocks() as upscale_block:
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    # Title
    upscale_title = gr.Markdown("# Upscale")

    with gr.Row():
        with gr.Column():
            with gr.Row():
                img_inp_upscale = gr.Image(
                    sources="upload", type="filepath"
                )
            with gr.Group():
                upscl = gr.Dropdown(
                    label="Upscaler",
                    choices=get_models(config.get('upscl_dir')),
                    value="",
                    allow_custom_value=True,
                    interactive=True
                )
                with gr.Row():
                    reload_btn = gr.Button(value=RELOAD_SYMBOL)
                    reload_btn.click(
                        reload_models,
                        inputs=[upscl_dir_txt],
                        outputs=[upscl]
                    )
                    gr.ClearButton(upscl)
            with gr.Group():
                rescale_bool = gr.Checkbox(
                    label="Allow changing initial size",
                    value=False
                )
                with gr.Row():
                    init_width = gr.Slider(
                        label="Initial Width",
                        minimum=1,
                        maximum=4096,
                        value=config.get('def_width'),
                        step=1,
                        show_reset_button=False,
                        interactive=False
                    )
                    init_height = gr.Slider(
                        label="Initial Height",
                        minimum=1,
                        maximum=4096,
                        value=config.get('def_height'),
                        step=1,
                        show_reset_button=False,
                        interactive=False
                    )
                switch_size_btn = gr.Button(
                    value=SWITCH_V_SYMBOL, scale=1,
                    interactive=False
                )
            switch_size_btn.click(
                switch_sizes,
                inputs=[init_height,
                        init_width],
                outputs=[init_height,
                         init_width]
            )
            upscl_rep = gr.Slider(
                label="Upscaler repeats",
                minimum=1,
                maximum=5,
                value=1,
                step=1
            )
            with gr.Row():
                with gr.Accordion(
                    label="Extra", open=False
                ):
                    output = gr.Textbox(
                        label="Output Name (optional)", value=""
                    )
                    color = gr.Checkbox(
                        label="Color", value=True
                    )
                    verbose = gr.Checkbox(label="Verbose")

        # Output
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    upscl_btn = gr.Button(
                        value="Upscale", size="lg",
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
                        label="Progress:",
                        visible=False,
                        interactive=False
                    )
                with gr.Row():
                    img_final = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        columns=[3],
                        rows=[1],
                        object_fit="contain",
                        height="auto",
                        interactive=False
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

    inputs_map = {
        'in_img_inp': img_inp_upscale,
        'in_upscl': upscl,
        'in_init_width': init_width,
        'in_init_height': init_height,
        'in_upscl_rep': upscl_rep,
        'in_output': output,
        'in_color': color,
        'in_verbose': verbose
    }

    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[key] for key in ordered_keys]

    def upscale_wrapper(*args):
        """
        Accepts all UI inputs, zips them with keys, and calls the
        main upscale function.
        """
        # This line programmatically creates the dictionary.
        params = dict(zip(ordered_keys, args))
        yield from upscale(params)

    def size_extractor(img_inp):
        try:
            with Image.open(img_inp) as img:
                width, height = img.size
        except Exception:
            width, height = None, None
        return (
            gr.update(value=int(width)), gr.update(value=int(height))
        )

    def rescale(rescale_bool):
        if rescale_bool:
            return(
                gr.update(interactive=True), gr.update(interactive=True),
                gr.update(interactive=True)
            )
        else:
            return(
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False)
            )

    # Interactive Bindings

    img_inp_upscale.change(
        size_extractor,
        inputs=img_inp_upscale,
        outputs=[init_width, init_height]
    )

    rescale_bool.change(
        rescale,
        inputs=rescale_bool,
        outputs=[init_width, init_height, switch_size_btn]
    )

    upscl_btn.click(
        upscale_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_slider, progress_textbox, stats, img_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

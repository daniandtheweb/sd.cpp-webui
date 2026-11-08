"""sd.cpp-webui - server tabs - upscale UI"""

import gradio as gr

from modules.core.server.sdcpp_server import upscale_api
from modules.loader import get_models, reload_models
from modules.shared_instance import config
from modules.ui.constants import RELOAD_SYMBOL
from modules.ui.progress import create_progress_bar
from modules.utils.ui_events import get_ordered_inputs, OUTPUT_GALLERY_CLASS


with gr.Blocks() as upscale_server_block:
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)

    gr.Markdown("# Upscale")
    gr.Markdown(
        "Uses the running server: start it from another tab first. The "
        "upscalers are the ones in the server's upscaler folder."
    )

    with gr.Row():
        with gr.Column():
            img_inp_upscale_server = gr.Image(
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
            upscl_rep = gr.Slider(
                label="Upscaler repeats",
                minimum=1,
                maximum=4,
                value=1,
                step=1
            )
            upscl_tile_size = gr.Number(
                label="Tile size for ESRGAN",
                minimum=1,
                maximum=4096,
                value=128,
                step=1,
            )
            with gr.Accordion(label="Extra", open=False):
                output = gr.Textbox(
                    label="Output Name (optional)", value=""
                )

        with gr.Column():
            with gr.Group():
                upscl_btn = gr.Button(
                    value="Upscale", size="lg", variant="primary"
                )
                progress_bar = create_progress_bar()
                img_final = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    columns=[3],
                    rows=[1],
                    object_fit="contain",
                    height="auto",
                    buttons=["share", "download", "fullscreen"],
                    elem_classes=[OUTPUT_GALLERY_CLASS],
                    interactive=False
                )
                stats = gr.Textbox(
                    label="Statistics:",
                    show_label=True,
                    value="",
                    interactive=False,
                    visible=False
                )
                command = gr.Textbox(
                    label="stable-diffusion.cpp request:",
                    show_label=True,
                    value="",
                    interactive=False,
                    buttons=['copy'],
                )

    inputs_map = {
        'in_img_inp': img_inp_upscale_server,
        'in_upscl': upscl,
        'in_upscl_rep': upscl_rep,
        'in_upscl_tile_size': upscl_tile_size,
        'in_output': output,
    }
    ordered_keys, ordered_components = get_ordered_inputs(inputs_map)

    def upscale_wrapper(*args):
        """Upscale with the UI inputs, on the running server."""
        yield from upscale_api(dict(zip(ordered_keys, args)))

    upscl_btn.click(
        upscale_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_bar, stats, img_final],
        show_progress_on=img_final
    )

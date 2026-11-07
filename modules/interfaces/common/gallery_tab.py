"""sd.cpp-webui - Gallery UI"""

import gradio as gr

from modules.extensions.host import (
    registered_gallery_actions, registered_outputs, run_gallery_action
)
from modules.extensions.ui import ensure_loaded
from modules.gallery import EXTENSION_CTRL_BASE, GalleryManager

from modules.shared_instance import config
from modules.ui.constants import (
    FIELDS, SORT_OPTIONS
)


ensure_loaded()
ext_outputs = registered_outputs()
ext_actions = registered_gallery_actions()

gallery_manager = GalleryManager()
info_params = {}


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(
        value=0, visible=False
    )
    img2img_ctrl = gr.Textbox(
        value=1, visible=False
    )
    imgedit_ctrl = gr.Textbox(
        value=2, visible=False
    )
    any2video_ctrl = gr.Textbox(
        value=3, visible=False
    )
    upscl_ctrl = gr.Textbox(
        value=4, visible=False
    )
    def_page = gr.Textbox(
        value=1, visible=False
    )

    # Title
    gallery_title = gr.Markdown('# Gallery')

    # Gallery Navigation Buttons
    with gr.Row():
        txt2img_btn = gr.Button(
            value="txt2img", variant="primary"
        )
        img2img_btn = gr.Button(
            value="img2img", variant="primary"
        )
        imgedit_btn = gr.Button(
            value="imgedit", variant="primary"
        )
        any2video_btn = gr.Button(
            value="any2video", variant="primary"
        )
        upscl_btn = gr.Button(
            value="upscale", variant="primary"
        )
        # One button per output folder an extension declared
        ext_btns = [
            gr.Button(value=label, variant="primary")
            for _, label, _ in ext_outputs
        ]

    with gr.Row():
        first_btn = gr.Button(value="First page")
        with gr.Group():
            page_num_select = gr.Number(
                label="Page",
                value=1,
                interactive=True,
                scale=7
            )
            go_btn = gr.Button(
                value="Go", scale=1
            )
        last_btn = gr.Button(value="Last page")

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    pvw_btn = gr.Button(value="Previous page")
                    nxt_btn = gr.Button(value="Next page")
            # Gallery Display
            gallery = gr.Gallery(
                label="txt2img",
                columns=[4],
                rows=[4],
                object_fit="contain",
                height="auto",
                fit_columns=False,
                scale=2
            )
            with gr.Row():
                sort_order = gr.Radio(
                    label="Sort By",
                    choices=SORT_OPTIONS,
                    value=config.get('def_gallery_sorting'),
                    interactive=True,
                )

        with gr.Column():
            # Positive prompts
            info_params['pprompt'] = gr.Textbox(
                label="Positive prompt",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                buttons=['copy'],
                max_lines=4
            )
            # Negative prompts
            info_params['nprompt'] = gr.Textbox(
                label="Negative prompt",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                buttons=['copy'],
                max_lines=4
            )
            with gr.Row():
                # Width
                info_params['width'] = gr.Number(
                    label="Width",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
                # Height
                info_params['height'] = gr.Number(
                    label="Height",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
            # Steps
            info_params['steps'] = gr.Number(
                label="Steps",
                value=None,
                interactive=False,
                scale=1,
                min_width=150
            )
            # Sampler
            info_params['sampling'] = gr.Textbox(
                label="Sampler",
                value="",
                interactive=False,
                scale=1,
                min_width=150,
                buttons=['copy'],
                max_lines=1
            )
            info_params['scheduler'] = gr.Textbox(
                label="Scheduler",
                value="",
                interactive=False,
                scale=1,
                min_width=150,
                buttons=['copy'],
                max_lines=1
            )
            with gr.Row():
                # CFG
                info_params['cfg'] = gr.Number(
                    label="CFG",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
                # Seed
                info_params['seed'] = gr.Number(
                    label="Seed",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
            # Path
            path_info = gr.Textbox(
                label="Path",
                value="",
                interactive=False,
                scale=1,
                min_width=150
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
            with gr.Row():
                # Copy to txt2img
                cpy_2_txt2img_btn = gr.Button(value="Copy to txt2img")
                # Copy to img2img
                cpy_2_img2img_btn = gr.Button(value="Copy to img2img")
                # Copy to imgedit
                cpy_2_imgedit_btn = gr.Button(value="Copy to imgedit")
                # Copy to any2video
                cpy_2_any2video_btn = gr.Button(value="Copy to any2video")
                # Copy to upscale
                cpy_2_upscale_btn = gr.Button(value="Copy to upscale")
            # Actions added by extensions, on the selected image
            action_btns = []
            if ext_actions:
                with gr.Row():
                    action_btns = [
                        gr.Button(value=label) for _, label, _ in ext_actions
                    ]
                action_status = gr.Textbox(
                    label="Extension action", interactive=False
                )
            # Delete image Button
            del_img = gr.Button(
                value="Delete", variant="stop")
            convert_to_mp4_btn = gr.Button(
                    value="Convert AVI to MP4",
                    interactive=False,
                    visible=False,
                    variant="primary"
                )

    param_ctrls = [info_params[f] for f in FIELDS]

    nav_outputs = [
        gallery, page_num_select
    ]

    # Interactive bindings
    gallery.select(
        gallery_manager.get_media_info,
        inputs=[],
        outputs=param_ctrls + [path_info, img_info_txt, convert_to_mp4_btn]
    )

    txt2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, txt2img_ctrl, sort_order],
        outputs=nav_outputs
    )

    img2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, img2img_ctrl, sort_order],
        outputs=nav_outputs
    )

    imgedit_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, imgedit_ctrl, sort_order],
        outputs=nav_outputs
    )

    any2video_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, any2video_ctrl, sort_order],
        outputs=nav_outputs
    )

    upscl_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, upscl_ctrl, sort_order],
        outputs=nav_outputs
    )

    for index, ext_btn in enumerate(ext_btns):
        ext_btn.click(
            gallery_manager.reload_gallery,
            inputs=[
                def_page, gr.State(EXTENSION_CTRL_BASE + index), sort_order
            ],
            outputs=nav_outputs
        )

    def make_action(ext, fn):
        # The selection is the server's, not a value the browser sends
        def run():
            return run_gallery_action(
                ext, fn, gallery_manager.current_media_path
            )
        return run

    for (ext, _, fn), action_btn in zip(ext_actions, action_btns):
        action_btn.click(
            make_action(ext, fn), inputs=[], outputs=[action_status]
        )

    pvw_btn.click(
        gallery_manager.prev_page,
        inputs=[],
        outputs=nav_outputs
    )

    nxt_btn.click(
        gallery_manager.next_page,
        inputs=[],
        outputs=nav_outputs
    )

    first_btn.click(
        gallery_manager.reload_gallery,
        inputs=[],
        outputs=nav_outputs
    )

    last_btn.click(
        gallery_manager.last_page,
        inputs=[],
        outputs=nav_outputs
    )

    go_btn.click(
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=nav_outputs
    )

    page_num_select.submit(
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=nav_outputs
    )

    sort_order.change(
        gallery_manager.reload_gallery,
        inputs=[page_num_select, gr.State(None), sort_order],
        outputs=nav_outputs
    )

    del_img.click(
        gallery_manager.delete_media,
        inputs=[],
        outputs=[gallery, page_num_select] + param_ctrls + [
                 path_info, img_info_txt, convert_to_mp4_btn]
    )

    convert_to_mp4_btn.click(
        gallery_manager.convert_to_mp4,
        inputs=[],
        outputs=nav_outputs
    )

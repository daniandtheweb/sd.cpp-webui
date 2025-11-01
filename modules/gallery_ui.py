"""sd.cpp-webui - Gallery UI"""

import gradio as gr

from modules.gallery import GalleryManager

from modules.shared_instance import config
from modules.ui.constants import FIELDS


gallery_manager = GalleryManager(
    config.get('txt2img_dir'),
    config.get('img2img_dir'),
    config.get('imgedit_dir'),
    config.get('any2video_dir'),
    config.get('upscale_dir')
)

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

    with gr.Group():
        with gr.Row():
            pvw_btn = gr.Button(value="Previous")
            nxt_btn = gr.Button(value="Next")

    with gr.Group():
        with gr.Row():
            first_btn = gr.Button(value="First page")
            last_btn = gr.Button(value="Last page")

    with gr.Group():
        with gr.Row():
            page_num_select = gr.Number(
                label="Page:",
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
            info_params['pprompt'] = gr.Textbox(
                label="Positive prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
            )
            # Negative prompts
            info_params['nprompt'] = gr.Textbox(
                label="Negative prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
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
                show_copy_button=True,
                max_lines=1
            )
            info_params['scheduler'] = gr.Textbox(
                label="Scheduler",
                value="",
                interactive=False,
                scale=1,
                min_width=150,
                show_copy_button=True,
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
            # Delete image Button
            del_img = gr.Button(
                value="Delete", variant="stop")

    param_ctrls = [info_params[f] for f in FIELDS]
    # Interactive bindings
    gallery.select(
        gallery_manager.get_img_info,
        inputs=[],
        outputs=param_ctrls + [path_info, img_info_txt]
    )
    txt2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, txt2img_ctrl],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    img2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, img2img_ctrl],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    imgedit_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, imgedit_ctrl],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    any2video_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, any2video_ctrl],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    upscl_btn.click(
        gallery_manager.reload_gallery,
        inputs=[def_page, upscl_ctrl],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    pvw_btn.click(
        gallery_manager.prev_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    nxt_btn.click(
        gallery_manager.next_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    first_btn.click(
        gallery_manager.reload_gallery,
        inputs=[],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    last_btn.click(
        gallery_manager.last_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    go_btn.click(
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    page_num_select.submit(
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery, gallery]
    )
    del_img.click(
        gallery_manager.delete_img,
        inputs=[],
        outputs=[gallery, page_num_select, gallery] + param_ctrls
                + [path_info, img_info_txt]
    )

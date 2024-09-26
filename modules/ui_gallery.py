"""sd.cpp-webui - Gallery UI"""

import gradio as gr

from modules.gallery import GalleryManager

from modules.config import (
    txt2img_dir, img2img_dir
)


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(
        value=0, visible=False
    )
    img2img_ctrl = gr.Textbox(
        value=1, visible=False
    )

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
        page_num_select = gr.Number(
            label="Page:",
            minimum=1,
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
            pprompt = gr.Textbox(
                label="Positive prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
            )
            # Negative prompts
            nprompt = gr.Textbox(
                label="Negative prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
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
            # Delete image Button
            del_img = gr.Button(value="Delete")

    # Interactive bindings
    gallery_manager = GalleryManager(
        txt2img_dir, img2img_dir
    )
    gallery.select(
        gallery_manager.img_info,
        inputs=[],
        outputs=[pprompt, nprompt, img_info_txt]
    )
    txt2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[txt2img_ctrl],
        outputs=[gallery, page_num_select, gallery]
    )
    img2img_btn.click(
        gallery_manager.reload_gallery,
        inputs=[img2img_ctrl],
        outputs=[gallery, page_num_select, gallery]
    )
    pvw_btn.click(
        gallery_manager.prev_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    nxt_btn.click(
        gallery_manager.next_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    first_btn.click(
        gallery_manager.reload_gallery,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    last_btn.click(
        gallery_manager.last_page,
        inputs=[],
        outputs=[gallery, page_num_select, gallery]
    )
    go_btn.click(
        gallery_manager.goto_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery]
    )
    del_img.click(
        gallery_manager.delete_img,
        inputs=[],
        outputs=[gallery, page_num_select, gallery,
                 pprompt, nprompt, img_info_txt]
    )

"""sd.cpp-webui - Gallery UI"""

import gradio as gr

from modules.gallery import GalleryManager

from modules.shared_instance import config


gallery_manager = GalleryManager(
    config.get('txt2img_dir'),
    config.get('img2img_dir'),
    config.get('any2video_dir')
)


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(
        value=0, visible=False
    )
    img2img_ctrl = gr.Textbox(
        value=1, visible=False
    )
    any2video_ctrl = gr.Textbox(
        value=2, visible=False
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
        any2video_btn = gr.Button(
            value="any2video", variant="primary"
        )

    with gr.Row():
        pvw_btn = gr.Button(value="Previous")
        nxt_btn = gr.Button(value="Next")

    with gr.Row():
        first_btn = gr.Button(value="First page")
        last_btn = gr.Button(value="Last page")

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
            pprompt_info = gr.Textbox(
                label="Positive prompt:",
                value="",
                interactive=False,
                scale=1,
                min_width=300,
                show_copy_button=True,
                max_lines=4
            )
            # Negative prompts
            nprompt_info = gr.Textbox(
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
                width_info = gr.Number(
                    label="Width",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
                # Height
                height_info = gr.Number(
                    label="Height",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
            # Steps
            steps_info = gr.Number(
                label="Steps",
                value=None,
                interactive=False,
                scale=1,
                min_width=150
            )
            # Sampler
            sampler_info = gr.Textbox(
                label="Sampler",
                value="",
                interactive=False,
                scale=1,
                min_width=150,
                show_copy_button=True,
                max_lines=1
            )
            scheduler_info = gr.Textbox(
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
                cfg_info = gr.Number(
                    label="CFG",
                    value=None,
                    interactive=False,
                    scale=1,
                    min_width=150
                )
                # Seed
                seed_info = gr.Number(
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
                # Copy to any2video
                cpy_2_any2video_btn = gr.Button(value="Copy to any2video")
            # Delete image Button
            del_img = gr.Button(
                value="Delete", variant="stop")

    # Interactive bindings
    gallery.select(
        gallery_manager.get_img_info,
        inputs=[],
        outputs=[pprompt_info, nprompt_info, width_info, height_info,
                 steps_info, sampler_info, scheduler_info, cfg_info,
                 seed_info, path_info, img_info_txt]
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
    any2video_btn.click(
        gallery_manager.reload_gallery,
        inputs=[any2video_ctrl],
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
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery]
    )
    page_num_select.submit(
        gallery_manager.reload_gallery,
        inputs=[page_num_select],
        outputs=[gallery, page_num_select, gallery]
    )
    del_img.click(
        gallery_manager.delete_img,
        inputs=[],
        outputs=[gallery, page_num_select, gallery,
                 pprompt_info, nprompt_info, width_info, height_info,
                 steps_info, sampler_info, scheduler_info, cfg_info,
                 seed_info, path_info, img_info_txt]
    )

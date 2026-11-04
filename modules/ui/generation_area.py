"""sd.cpp-webui - UI component for the generate area of the tabs"""

from typing import NamedTuple

import gradio as gr

from modules.utils.ui_events import OUTPUT_GALLERY_CLASS
from .progress import create_progress_bar


class GenerationArea(NamedTuple):
    gen_btn: gr.Button
    kill_btn: gr.Button
    queue_tracker: gr.Textbox
    progress_bar: gr.Progress
    gallery: gr.Gallery
    stats: gr.Textbox
    command: gr.Textbox


def create_generation_area(video: bool = False, server: bool = False):
    """
    Create the generate and stop buttons, the progress and the results.
    The server tabs show the request payload instead of the command, and
    start with the generate button disabled until the server runs.
    """
    with gr.Group():
        with gr.Row():
            gen_btn = gr.Button(
                value="Generate", size="lg",
                variant="primary", interactive=not server
            )
            kill_btn = gr.Button(
                value="Stop", size="lg",
                variant="stop"
            )
        with gr.Row():
            queue_tracker = gr.Textbox(
                show_label=False,
                visible=False
            )
        with gr.Row():
            progress_bar = create_progress_bar()
        with gr.Row():
            gallery = gr.Gallery(
                label="Generated videos" if video else "Generated images",
                show_label=False,
                columns=[3],
                rows=[1],
                object_fit="contain",
                height="auto",
                buttons=["share", "download", "fullscreen"],
                elem_classes=[OUTPUT_GALLERY_CLASS]
            )
        with gr.Row():
            stats = gr.Textbox(
                label="Statistics:",
                show_label=True,
                value="",
                interactive=False,
                visible=False
            )
        with gr.Row():
            command = gr.Textbox(
                label="Request payload:" if server
                else "stable-diffusion.cpp command:",
                show_label=True,
                value="",
                interactive=False,
                buttons=['copy'],
            )

    return GenerationArea(
        gen_btn, kill_btn, queue_tracker, progress_bar, gallery, stats,
        command
    )

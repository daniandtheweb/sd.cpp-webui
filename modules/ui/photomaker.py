"""sd.cpp-webui - UI component for photomaker"""

from functools import partial

import gradio as gr

from modules.loader import (
    get_models, reload_models
)
from modules.shared_instance import config
from modules.utils.ui_handler import update_interactivity
from .constants import RELOAD_SYMBOL

def create_photomaker_ui():
    # Directory Textbox
    phtmkr_dir_txt = gr.Textbox(value=config.get('phtmkr_dir'), visible=False)

    # PhotoMaker
    with gr.Row():
        with gr.Group():
            with gr.Row():
                phtmkr_bool = gr.Checkbox(
                    label="Enable PhotoMaker", value=False
                )
            with gr.Row():
                phtmkr = gr.Dropdown(
                    label="PhotoMaker",
                    choices=get_models(config.get('phtmkr_dir')),
                    value="",
                    allow_custom_value=True,
                    interactive=False
                )
            with gr.Row():
                reload_phtmkr_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=False
                )
                clear_phtmkr_btn = gr.ClearButton(
                    phtmkr,
                    interactive=False
                )
            with gr.Row():
                phtmkr_id = gr.Textbox(
                    label="PhotoMaker input id images directory",
                    value="",
                    interactive=False
                        )
            with gr.Row():
                phtmkr_emb = gr.Textbox(
                    label="PhotoMaker v2 id embed",
                    value="",
                    interactive=False
                    )
            with gr.Row():
                phtmkr_strength = gr.Slider(
                    label="PhotoMaker style strength",
                    value=20,
                    interactive=False,
                    minimum=0,
                    maximum=100,
                    step=1
                )

    phtmkr_comp = [
        phtmkr, reload_phtmkr_btn, clear_phtmkr_btn,
        phtmkr_id, phtmkr_emb, phtmkr_strength
    ]

    reload_phtmkr_btn.click(
        reload_models,
        inputs=[phtmkr_dir_txt],
        outputs=[phtmkr]
    )
    phtmkr_bool.change(
        partial(update_interactivity, len(phtmkr_comp)),
        inputs=phtmkr_bool,
        outputs=phtmkr_comp
    )

    return {
        'in_phtmkr_bool': phtmkr_bool,
        'in_phtmkr': phtmkr,
        'in_phtmkr_id': phtmkr_id,
        'in_phtmkr_emb': phtmkr_emb,
        'in_phtmkr_strength': phtmkr_strength
    }

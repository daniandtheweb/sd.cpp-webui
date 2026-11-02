"""sd.cpp-webui - UI component for PuLID"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.ui_events import update_interactivity
from .constants import RELOAD_SYMBOL, PULID_EMBEDDING_EXT


def create_pulid_ui():
    """Create the PuLID UI"""
    pulid_dir_txt = gr.Textbox(value=config.get('pulid_dir'), visible=False)

    with gr.Accordion(
        label="PuLID", open=False
    ):
        pulid_bool = gr.Checkbox(
            label="Enable PuLID", value=False
        )
        with gr.Group():
            pulid_weights = gr.Dropdown(
                label="PuLID weights (Flux, v0.9.x)",
                choices=get_models(config.get('pulid_dir')),
                value=None,
                interactive=False
            )
            with gr.Row():
                reload_pulid_weights_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=False
                )
                clear_pulid_weights_btn = gr.ClearButton(
                    pulid_weights,
                    interactive=False
                )
        with gr.Group():
            pulid_embedding = gr.Dropdown(
                label="ID embedding (.pulidembd)",
                choices=get_models(
                    config.get('pulid_dir'), PULID_EMBEDDING_EXT
                ),
                value=None,
                info="Precomputed with sd.cpp's pulid_extract_id.py",
                interactive=False
            )
            with gr.Row():
                reload_pulid_embedding_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=False
                )
                clear_pulid_embedding_btn = gr.ClearButton(
                    pulid_embedding,
                    interactive=False
                )
        pulid_id_weight = gr.Slider(
            label="PuLID identity weight",
            minimum=0,
            maximum=2,
            step=0.01,
            value=1.0,
            info="Typical range 0.7 to 1.2. Not compatible with "
                 "Skip Layer Guidance",
            interactive=False
        )

    pulid_comp = [
        pulid_weights, reload_pulid_weights_btn, clear_pulid_weights_btn,
        pulid_embedding, reload_pulid_embedding_btn,
        clear_pulid_embedding_btn, pulid_id_weight
    ]

    reload_pulid_weights_btn.click(
        reload_models,
        inputs=[pulid_dir_txt],
        outputs=[pulid_weights]
    )

    reload_pulid_embedding_btn.click(
        partial(reload_models, extensions=PULID_EMBEDDING_EXT),
        inputs=[pulid_dir_txt],
        outputs=[pulid_embedding]
    )

    pulid_bool.change(
        partial(update_interactivity, len(pulid_comp)),
        inputs=pulid_bool,
        outputs=pulid_comp
    )

    return {
        'in_pulid_bool': pulid_bool,
        'in_pulid_weights': pulid_weights,
        'in_pulid_embedding': pulid_embedding,
        'in_pulid_id_weight': pulid_id_weight,
    }

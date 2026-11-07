"""sd.cpp-webui - UI component for ControlNet"""

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.image_utils import size_extractor
from modules.utils.ui_events import update_interactivity
from .constants import CONTROL_TYPES, RELOAD_SYMBOL


def create_cnnet_ui(model_sel: bool = True, inputs: bool = True):
    """
    Create the ControlNet UI
    model_sel: show the model selection (a launch option in server mode)
    inputs: show the enable checkbox, control image and strength
    """
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    # Without the enable checkbox the model selection is always active
    model_interactive = not inputs

    with gr.Accordion(
        label="ControlNet", open=False
    ):
        cnnet_bool = gr.Checkbox(
            label="Enable ControlNet", value=False,
            visible=inputs
        )
        with gr.Group(visible=model_sel):
            cnnet = gr.Dropdown(
                label="ControlNet",
                choices=get_models(config.get('cnnet_dir')),
                value=None,
                interactive=model_interactive
            )
            with gr.Row():
                reload_cnnet_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=model_interactive)
                clear_cnnet_btn = gr.ClearButton(
                    cnnet,
                    interactive=model_interactive
                )
        control_img = gr.Image(
            sources="upload", type="filepath",
            interactive=False,
            visible=inputs
        )
        control_info = gr.Markdown(visible=inputs)
        control_strength = gr.Slider(
            label="ControlNet strength",
            minimum=0,
            maximum=1,
            step=0.01,
            value=0.9,
            interactive=False,
            visible=inputs
        )
        cnnet_union = gr.Checkbox(
            label="ControlNet Union", value=False,
            info="The model is a ControlNet Union: it needs to be told "
                 "the kind of the control image.",
            interactive=False,
            visible=inputs
        )
        control_type = gr.Dropdown(
            label="Control type",
            choices=CONTROL_TYPES,
            value=CONTROL_TYPES[0] if CONTROL_TYPES else None,
            interactive=False,
            visible=inputs
        )

    cnnet_comp = [
        cnnet, reload_cnnet_btn, clear_cnnet_btn, control_img,
        control_strength, cnnet_union
    ]

    def set_cnnet_interactivity(enabled, union):
        """
        ControlNet switches its widgets on; the control type also needs the
        union checkbox.
        """
        return [
            *update_interactivity(len(cnnet_comp), enabled),
            gr.update(interactive=bool(enabled and union))
        ]

    def describe_control_img(path):
        """The size of the control image, shown under it."""
        if not path:
            return ""
        width, height = size_extractor(path)
        if not width:
            return ""
        return f"Control image size: **{width} x {height}**"

    control_img.change(
        describe_control_img, inputs=[control_img], outputs=[control_info]
    )

    reload_cnnet_btn.click(
        reload_models,
        inputs=[cnnet_dir_txt],
        outputs=[cnnet]
    )

    cnnet_bool.change(
        set_cnnet_interactivity,
        inputs=[cnnet_bool, cnnet_union],
        outputs=cnnet_comp + [control_type]
    )

    cnnet_union.change(
        lambda enabled, union: gr.update(
            interactive=bool(enabled and union)
        ),
        inputs=[cnnet_bool, cnnet_union],
        outputs=control_type
    )

    return {
        'in_cnnet_bool': cnnet_bool,
        'in_cnnet': cnnet,
        'in_control_img': control_img,
        'in_control_strength': control_strength,
        'in_cnnet_union': cnnet_union,
        'in_control_type': control_type
    }

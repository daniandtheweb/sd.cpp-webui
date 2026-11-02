"""sd.cpp-webui - UI component for IP-Adapter"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.ui_events import update_interactivity
from .constants import RELOAD_SYMBOL


def create_ipadapter_ui(model_sel: bool = True, inputs: bool = True):
    """
    Create the IP-Adapter UI
    model_sel: show the IP-Adapter and CLIP-Vision selection
               (launch options in server mode)
    inputs: show the enable checkbox, reference image and strength
    """
    ipadapter_dir_txt = gr.Textbox(
        value=config.get('ipadapter_dir'), visible=False
    )
    txt_enc_dir_txt = gr.Textbox(
        value=config.get('txt_enc_dir'), visible=False
    )

    # Without the enable checkbox the model selection is always active
    model_interactive = not inputs

    with gr.Accordion(
        label="IP-Adapter", open=False
    ):
        ip_adapter_bool = gr.Checkbox(
            label="Enable IP-Adapter", value=False,
            visible=inputs
        )
        with gr.Group(visible=model_sel):
            ip_adapter = gr.Dropdown(
                label="IP-Adapter (SD 1.5 / SDXL)",
                choices=get_models(config.get('ipadapter_dir')),
                value=None,
                interactive=model_interactive
            )
            with gr.Row():
                reload_ip_adapter_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=model_interactive
                )
                clear_ip_adapter_btn = gr.ClearButton(
                    ip_adapter,
                    interactive=model_interactive
                )
        with gr.Group(visible=model_sel):
            ip_adapter_clip_vision = gr.Dropdown(
                label="CLIP-Vision (ViT-H/14)",
                choices=get_models(config.get('txt_enc_dir')),
                value=config.get('def_clip_vision_h'),
                info="text_encoders folder",
                interactive=model_interactive
            )
            with gr.Row():
                reload_clip_vision_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=model_interactive
                )
                clear_clip_vision_btn = gr.ClearButton(
                    ip_adapter_clip_vision,
                    interactive=model_interactive
                )
        ip_adapter_img = gr.Image(
            label="Reference image",
            sources="upload", type="filepath",
            interactive=False,
            visible=inputs
        )
        ip_adapter_strength = gr.Slider(
            label="IP-Adapter strength",
            minimum=0,
            maximum=2,
            step=0.01,
            value=1.0,
            info="Lower values let the prompt dominate, "
                 "0.6 to 0.8 is a good starting range",
            interactive=False,
            visible=inputs
        )

    ip_adapter_comp = [
        ip_adapter, reload_ip_adapter_btn, clear_ip_adapter_btn,
        ip_adapter_clip_vision, reload_clip_vision_btn,
        clear_clip_vision_btn, ip_adapter_img, ip_adapter_strength
    ]

    reload_ip_adapter_btn.click(
        reload_models,
        inputs=[ipadapter_dir_txt],
        outputs=[ip_adapter]
    )

    reload_clip_vision_btn.click(
        reload_models,
        inputs=[txt_enc_dir_txt],
        outputs=[ip_adapter_clip_vision]
    )

    ip_adapter_bool.change(
        partial(update_interactivity, len(ip_adapter_comp)),
        inputs=ip_adapter_bool,
        outputs=ip_adapter_comp
    )

    return {
        'in_ip_adapter_bool': ip_adapter_bool,
        'in_ip_adapter': ip_adapter,
        'in_ip_adapter_clip_vision': ip_adapter_clip_vision,
        'in_ip_adapter_img': ip_adapter_img,
        'in_ip_adapter_strength': ip_adapter_strength,
    }

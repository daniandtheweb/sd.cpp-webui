"""sd.cpp-webui - UI component for ADetailer"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.loader import (
    get_models, reload_models
)
from modules.utils.ui_events import update_interactivity
from .constants import RELOAD_SYMBOL, AD_MASK_MODES


def create_adetailer_ui(model_sel: bool = True, inputs: bool = True):
    """
    Create the ADetailer UI
    model_sel: show the detector selection (a launch option in server mode)
    inputs: show the prompts and detail pass settings
    With both, the enable checkbox controls everything. In server mode
    the detector runs on every image once loaded, so the split parts have
    no checkbox and are always active.
    """
    ad_dir_txt = gr.Textbox(value=config.get('ad_dir'), visible=False)

    toggle = model_sel and inputs
    interactive = not toggle

    with gr.Accordion(
        label="ADetailer", open=False
    ):
        ad_bool = gr.Checkbox(
            label="Enable ADetailer", value=False,
            visible=toggle
        )
        with gr.Group(visible=model_sel):
            ad_model = gr.Dropdown(
                label="Detector model (converted YOLOv8)",
                choices=get_models(config.get('ad_dir')),
                value=None,
                interactive=interactive
            )
            with gr.Row():
                reload_ad_btn = gr.Button(
                    value=RELOAD_SYMBOL,
                    interactive=interactive
                )
                clear_ad_btn = gr.ClearButton(
                    ad_model,
                    interactive=interactive
                )
        with gr.Group(visible=inputs):
            ad_prompt = gr.Textbox(
                label="ADetailer prompt",
                placeholder="Empty uses the main prompt",
                info=("Supports [PROMPT], [SEP] and [SKIP]"
                      if toggle else
                      "Supports [PROMPT], [SEP] and [SKIP]. Applied to "
                      "every image when a detector is loaded in the "
                      "Server Settings"),
                value=config.get('def_ad_prompt'),
                lines=2,
                interactive=interactive
            )
            ad_nprompt = gr.Textbox(
                label="ADetailer negative prompt",
                placeholder="Empty uses the main negative prompt",
                info="Supports [PROMPT] and [SEP]",
                value=config.get('def_ad_nprompt'),
                lines=2,
                interactive=interactive
            )
        with gr.Group(visible=inputs):
            with gr.Row():
                ad_confidence = gr.Slider(
                    label="Detection confidence",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=config.get('def_ad_confidence'),
                    interactive=interactive
                )
                ad_denoising_strength = gr.Slider(
                    label="Denoising strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=config.get('def_ad_denoising_strength'),
                    interactive=interactive
                )
            with gr.Row():
                ad_mask_blur = gr.Number(
                    label="Mask blur",
                    minimum=0,
                    precision=0,
                    step=1,
                    value=config.get('def_ad_mask_blur'),
                    interactive=interactive
                )
                ad_inpaint_padding = gr.Number(
                    label="Inpaint padding",
                    minimum=0,
                    precision=0,
                    step=1,
                    value=config.get('def_ad_inpaint_padding'),
                    interactive=interactive
                )
                ad_dilate_erode = gr.Number(
                    label="Mask dilate (+) / erode (-)",
                    precision=0,
                    step=1,
                    value=config.get('def_ad_dilate_erode'),
                    interactive=interactive
                )
            ad_mask_mode = gr.Dropdown(
                label="Mask mode",
                choices=AD_MASK_MODES,
                value=config.get('def_ad_mask_mode'),
                info="merge repairs all the detections in a single pass",
                interactive=interactive
            )
            with gr.Row():
                ad_inpaint_width = gr.Number(
                    label="Inpaint width",
                    minimum=1,
                    maximum=4096,
                    precision=0,
                    step=1,
                    value=config.get('def_ad_inpaint_width'),
                    interactive=interactive
                )
                ad_inpaint_height = gr.Number(
                    label="Inpaint height",
                    minimum=1,
                    maximum=4096,
                    precision=0,
                    step=1,
                    value=config.get('def_ad_inpaint_height'),
                    interactive=interactive
                )
            with gr.Row():
                ad_steps = gr.Number(
                    label="Steps (0 reuses steps)",
                    minimum=0,
                    maximum=100,
                    precision=0,
                    step=1,
                    value=config.get('def_ad_steps'),
                    interactive=interactive
                )
                ad_cfg = gr.Number(
                    label="CFG scale (-1 reuses CFG)",
                    minimum=-1,
                    precision=1,
                    step=0.5,
                    value=config.get('def_ad_cfg'),
                    interactive=interactive
                )
            ad_extra_args = gr.Textbox(
                label="Extra arguments (key=value, comma-separated)",
                placeholder="e.g. mask_k_largest=1,sort_by=area",
                value=config.get('def_ad_extra_args'),
                interactive=interactive
            )

    ad_comp = [
        ad_model, reload_ad_btn, clear_ad_btn, ad_prompt, ad_nprompt,
        ad_confidence, ad_denoising_strength, ad_mask_blur,
        ad_inpaint_padding, ad_dilate_erode, ad_mask_mode,
        ad_inpaint_width, ad_inpaint_height, ad_steps, ad_cfg,
        ad_extra_args
    ]

    reload_ad_btn.click(
        reload_models,
        inputs=[ad_dir_txt],
        outputs=[ad_model]
    )

    ad_bool.change(
        partial(update_interactivity, len(ad_comp)),
        inputs=ad_bool,
        outputs=ad_comp
    )

    return {
        'in_ad_bool': ad_bool,
        'in_ad_model': ad_model,
        'in_ad_prompt': ad_prompt,
        'in_ad_nprompt': ad_nprompt,
        'in_ad_confidence': ad_confidence,
        'in_ad_denoising_strength': ad_denoising_strength,
        'in_ad_mask_blur': ad_mask_blur,
        'in_ad_inpaint_padding': ad_inpaint_padding,
        'in_ad_dilate_erode': ad_dilate_erode,
        'in_ad_mask_mode': ad_mask_mode,
        'in_ad_inpaint_width': ad_inpaint_width,
        'in_ad_inpaint_height': ad_inpaint_height,
        'in_ad_steps': ad_steps,
        'in_ad_cfg': ad_cfg,
        'in_ad_extra_args': ad_extra_args,
    }

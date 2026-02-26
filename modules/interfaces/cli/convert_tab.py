"""sd.cpp-webui - Convert UI"""

import gradio as gr

from modules.core.cli.sdcpp_cli import convert
from modules.shared_instance import (
    config, subprocess_manager
)
from modules.loader import (
    get_models, reload_models, model_choice
)
from modules.ui.constants import (
    MODELS, QUANTS, RELOAD_SYMBOL
)


with gr.Blocks() as convert_block:
    inputs_map = {}

    ckpt_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    txt_enc_dir_txt = gr.Textbox(value=config.get('txt_enc_dir'), visible=False)
    emb_dir_txt = gr.Textbox(value=config.get('emb_dir'), visible=False)
    lora_dir_txt = gr.Textbox(value=config.get('lora_dir'), visible=False)
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    # Active model directory holder
    model_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
    inputs_map['in_model_dir'] = model_dir_txt

    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(
                label="Model Type",
                choices=MODELS,
                interactive=True,
                value="Checkpoint"
            )
            model_type.input(
                model_choice,
                inputs=[model_type],
                outputs=[model_dir_txt]
            )

            with gr.Group():
                orig_model = gr.Dropdown(
                    label="Model",
                    choices=get_models(config.get('ckpt_dir')),
                    scale=5,
                    interactive=True
                )
                reload_btn = gr.Button(
                    RELOAD_SYMBOL, scale=1
                )
                reload_btn.click(
                    reload_models,
                    inputs=[model_dir_txt],
                    outputs=[orig_model]
                )
                inputs_map['in_orig_model'] = orig_model

            with gr.Row():
                gguf_name = gr.Textbox(
                    label="Output Name (optional, must end with .gguf)",
                    value=""
                )
                inputs_map['in_gguf_name'] = gguf_name

            with gr.Row():
                quant_type = gr.Dropdown(
                    label="Type",
                    choices=QUANTS,
                    value=QUANTS[0],
                    interactive=True
                )
                inputs_map['in_quant_type'] = quant_type

            with gr.Accordion(
                label="Weight type per tensor pattern",
                open=False
            ):
                tensor_type_rules = gr.Textbox(
                    show_label=False,
                    container=False,
                    value="",
                    placeholder="example: \"^vae\\.=f16,model\\.=q8_0\"",
                    interactive=True
                )
                inputs_map['in_tensor_type_rules'] = tensor_type_rules

            color = gr.Checkbox(
                label="Color", value=True
            )
            inputs_map['in_color'] = color

            verbose = gr.Checkbox(label="Verbose")
            inputs_map['in_verbose'] = verbose

        with gr.Column():
            with gr.Group():
                with gr.Row():
                    convert_btn = gr.Button(
                        value="Convert", variant="primary"
                    )
                    kill_btn = gr.Button(
                        value="Stop", variant="stop"
                    )
                with gr.Row():
                    progress_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False,
                        visible=False,
                        label="Progress",
                        show_reset_button=False
                    )
                with gr.Row():
                    status_textbox = gr.Textbox(
                        label="Status:",
                        visible=False,
                        interactive=False
                    )

                with gr.Row():
                    command = gr.Textbox(
                        label="stable-diffusion.cpp command:",
                        show_label=True,
                        value="",
                        interactive=False,
                        show_copy_button=True,
                    )

    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[key] for key in ordered_keys]

    def convert_wrapper(*args):
        """
        Accepts all UI inputs, zips them with keys, and calls the
        main convert function.
        """
        params = dict(zip(ordered_keys, args))
        yield from convert(params)

    # Interactive Bindings
    convert_btn.click(
        convert_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_slider, status_textbox]
    )

    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    model_dir_txt.change(
        reload_models,
        inputs=[model_dir_txt],
        outputs=[orig_model]
    )

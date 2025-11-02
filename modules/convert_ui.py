"""sd.cpp-webui - Convert UI"""

import gradio as gr

from modules.sdcpp import convert
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
    ckpt_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
    vae_dir_txt = gr.Textbox(value=config.get('vae_dir'), visible=False)
    unet_dir_txt = gr.Textbox(value=config.get('unet_dir'), visible=False)
    clip_dir_txt = gr.Textbox(value=config.get('clip_dir'), visible=False)
    emb_dir_txt = gr.Textbox(value=config.get('emb_dir'), visible=False)
    lora_dir_txt = gr.Textbox(value=config.get('lora_dir'), visible=False)
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)
    model_dir_txt = gr.Textbox(value=config.get('ckpt_dir'), visible=False)
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

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    model = gr.Dropdown(
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
                        outputs=[model]
                    )
            with gr.Row():
                gguf_name = gr.Textbox(
                    label="Output Name (optional, must end with .gguf)",
                    value=""
                )

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    quant_type = gr.Dropdown(
                        label="Type",
                        choices=QUANTS,
                        value=QUANTS[0],
                        interactive=True
                    )
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

            verbose = gr.Checkbox(label="Verbose")

            with gr.Row():
                convert_btn = gr.Button(
                    value="Convert", variant="primary"
                )
                kill_btn = gr.Button(
                    value="Stop", variant="stop"
                )

        # Output
        with gr.Column(scale=1):
            result = gr.Textbox(
                interactive=False,
                value="",
                label="LOG"
            )

    # Interactive Bindings
    convert_btn.click(
        convert,
        inputs=[model, model_dir_txt, quant_type, tensor_type_rules,
                gguf_name, verbose],
        outputs=[result]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    model_dir_txt.change(
        reload_models,
        inputs=[model_dir_txt],
        outputs=[model]
    )

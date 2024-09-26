"""sd.cpp-webui - Convert UI"""

import gradio as gr

from modules.sdcpp import convert
from modules.utility import kill_subprocess
from modules.config import (
    sd_dir, vae_dir, flux_dir, clip_l_dir, t5xxl_dir, emb_dir,
    lora_dir, taesd_dir, upscl_dir, cnnet_dir
)
from modules.loader import (
    get_models, reload_models, model_choice, model_dir
)

QUANTS = ["Default", "f32", "f16", "q8_0", "q4_k", "q3_k", "q2_k", "q5_1",
          "q5_0", "q4_1", "q4_0"]
MODELS = ["Stable-Diffusion", "FLUX", "VAE", "clip_l", "t5xxl", "TAESD",
          "Lora", "Embeddings", "Upscaler", "ControlNet"]
RELOAD_SYMBOL = '\U0001f504'


with gr.Blocks() as convert_block:
    sd_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    flux_dir_txt = gr.Textbox(value=flux_dir, visible=False)
    clip_l_dir_txt = gr.Textbox(value=clip_l_dir, visible=False)
    t5xxl_dir_txt = gr.Textbox(value=t5xxl_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)
    model_dir_txt = gr.Textbox(value=sd_dir, visible=False)
    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        with gr.Column():
            model_type = gr.Dropdown(
                label="Model Type",
                choices=MODELS,
                interactive=True,
                value="Stable-Diffusion"
            )
            model_type.input(
                model_choice,
                inputs=[model_type],
                outputs=[model_dir_txt]
            )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                model = gr.Dropdown(
                    label="Model",
                    choices=get_models(model_dir),
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
                quant_type = gr.Dropdown(
                    label="Type",
                    choices=QUANTS,
                    value="f32",
                    interactive=True
                )

            verbose = gr.Checkbox(label="Verbose")

            with gr.Row():
                convert_btn = gr.Button(value="Convert")
                kill_btn = gr.Button(value="Stop")

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
        inputs=[model, model_dir_txt, quant_type,
                gguf_name, verbose],
        outputs=[result]
    )
    kill_btn.click(
        kill_subprocess,
        inputs=[],
        outputs=[]
    )

    model_dir_txt.change(
        reload_models,
        inputs=[model_dir_txt],
        outputs=[model]
    )

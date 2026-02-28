"""sd.cpp-webui - Image edit UI"""

import gradio as gr

from modules.core.cli.sdcpp_cli import imgedit
from modules.utils.ui_handler import (
    get_ordered_inputs, bind_generation_pipeline,
    refresh_all_options
)
from modules.shared_instance import (
    config, subprocess_manager
)
from modules.ui.models import create_imgedit_model_sel_ui
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.qwen import create_qwen_ui
from modules.ui.circular import create_circular_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.eta import create_eta_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.preview import create_preview_ui
from modules.ui.performance import create_performance_ui
from modules.ui.environment import create_env_ui
# from modules.ui.experimental import create_experimental_ui


with gr.Blocks() as imgedit_block:
    inputs_map = {}
    diffusion_mode = gr.Number(value=1, visible=False)
    # Directory Textboxes
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)

    # Title
    imgedit_title = gr.Markdown("# Image Edit")

    with gr.Accordion(
        label="Models selection", open=False
    ):
        # Model & VAE Selection
        model_ui = create_imgedit_model_sel_ui()
        inputs_map.update(model_ui)
        inputs_map['in_diffusion_mode'] = diffusion_mode

        # Model Type Selection
        quant_ui = create_quant_ui()
        inputs_map.update(quant_ui)

    # Prompts
    prompts_ui = create_prompts_ui(nprompt_support=False)
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            with gr.Tab("Generation Settings"):

                generation_settings_ui = create_generation_settings_ui(unet_mode=True)
                inputs_map.update(generation_settings_ui)

                bottom_generation_settings_ui = create_bottom_generation_settings_ui()
                inputs_map.update(bottom_generation_settings_ui)

            with gr.Tab("Image Enhancement"):

                # Upscale
                upscl_ui = create_upscl_ui()
                inputs_map.update(upscl_ui)

                # ControlNet
                cnnet_ui = create_cnnet_ui()
                inputs_map.update(cnnet_ui)

                # Qwen
                qwen_ui = create_qwen_ui()
                inputs_map.update(qwen_ui)

                # Circular padding
                circular_ui = create_circular_ui()
                inputs_map.update(circular_ui)

                # PhotoMaker
                photomaker_ui = create_photomaker_ui()
                inputs_map.update(photomaker_ui)

                # ETA
                eta_ui = create_eta_ui()
                inputs_map.update(eta_ui)

            with gr.Tab("Advanced Settings"):

                # TAESD
                taesd_ui = create_taesd_ui()
                inputs_map.update(taesd_ui)

                # VAE Tiling
                vae_tiling_ui = create_vae_tiling_ui()
                inputs_map.update(vae_tiling_ui)

                # Cache
                cache_ui = create_cache_ui()
                inputs_map.update(cache_ui)

                # Extra Settings
                extras_ui = create_extras_ui()
                inputs_map.update(extras_ui)

                # Preview Settings
                preview_ui = create_preview_ui()
                inputs_map.update(preview_ui)

                # Performance Settings
                performance_ui = create_performance_ui()
                inputs_map.update(performance_ui)

                # Environment Variables
                env_ui = create_env_ui()
                inputs_map.update(env_ui)

            # Experimental
            # experimental_ui = create_experimental_ui()
            # inputs_map.update(experimental_ui)

            with gr.Row():
                refresh_opt = gr.Button(
                    value="Refresh sd options"
                )

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                ref_img_imgedit = gr.Image(
                    sources="upload", type="filepath"
                )
                inputs_map['in_ref_img'] = ref_img_imgedit
            with gr.Group():
                with gr.Row():
                    gen_btn = gr.Button(
                        value="Generate", size="lg",
                        variant="primary"
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
                    progress_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False,
                        visible=False,
                        label="Progress",
                    )
                with gr.Row():
                    progress_textbox = gr.Textbox(
                        label="Status:",
                        visible=False,
                        interactive=False
                    )
                with gr.Row():
                    img_final = gr.Gallery(
                        label="Generated images",
                        show_label=False,
                        columns=[3],
                        rows=[1],
                        object_fit="contain",
                        height="auto"
                    )
                with gr.Row():
                    stats = gr.Textbox(
                        label="Statistics:",
                        show_label=True,
                        value="",
                        interactive=False
                    )
                with gr.Row():
                    command = gr.Textbox(
                        label="stable-diffusion.cpp command:",
                        show_label=True,
                        value="",
                        interactive=False,
                        buttons=['copy'],
                    )

    ordered_keys, ordered_components = get_ordered_inputs(inputs_map)

    timer = gr.Timer(value=0.01, active=True)

    ui_outputs = {
        'gen_btn': gen_btn,
        'timer': timer,
        'command': command,
        'progress_slider': progress_slider,
        'progress_textbox': progress_textbox,
        'stats': stats,
        'img_final': img_final,
        'queue_tracker': queue_tracker
    }

    bind_generation_pipeline(
        imgedit, ordered_keys, ordered_components, ui_outputs
    )

    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            preview_ui['in_preview_mode'], extras_ui['in_predict']
        ]
    )

    width_imgedit = generation_settings_ui['in_width']
    height_imgedit = generation_settings_ui['in_height']

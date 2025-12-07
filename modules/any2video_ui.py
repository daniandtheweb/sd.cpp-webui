"""sd.cpp-webui - Anything to Video UI"""

from functools import partial

import gradio as gr

from modules.sdcpp import any2video
from modules.utils.ui_handler import (
    update_interactivity, refresh_all_options
)
from modules.shared_instance import (
    config, subprocess_manager
)
from modules.ui.models import create_video_model_sel_ui
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.eta import create_eta_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.easycache import create_easycache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.preview import create_preview_ui
from modules.ui.performance import create_performance_ui
from modules.ui.environment import create_env_ui
# from modules.ui.experimental import create_experimental_ui

any2video_params = {}

with gr.Blocks() as any2video_block:
    inputs_map = {}
    # Directory Textboxes
    emb_dir_txt = gr.Textbox(value=config.get('emb_dir'), visible=False)
    lora_dir_txt = gr.Textbox(value=config.get('lora_dir'), visible=False)
    taesd_dir_txt = gr.Textbox(value=config.get('taesd_dir'), visible=False)
    phtmkr_dir_txt = gr.Textbox(value=config.get('phtmkr_dir'), visible=False)
    upscl_dir_txt = gr.Textbox(value=config.get('upscl_dir'), visible=False)
    cnnet_dir_txt = gr.Textbox(value=config.get('cnnet_dir'), visible=False)

    # Title
    any2video_title = gr.Markdown("# Anything to Video")

    with gr.Accordion(
        label="Models selection", open=False
    ):
        # Model & VAE Selection
        model_ui = create_video_model_sel_ui()
        inputs_map.update(model_ui)

        # Model Type Selection
        quant_ui = create_quant_ui()
        inputs_map.update(quant_ui)

    # Prompts
    prompts_ui = create_prompts_ui()
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            with gr.Tab("Generation Settings"):

                generation_settings_ui = create_generation_settings_ui()
                inputs_map.update(generation_settings_ui)

                with gr.Row():
                    frames = gr.Number(
                        label="Video Frames",
                        minimum=1,
                        value=1,
                        scale=1,
                        interactive=True,
                        step=1
                    )
                    fps = gr.Number(
                        label="FPS",
                        minimum=1,
                        value=24,
                        scale=1,
                        interactive=True,
                        step=1
                    )
                inputs_map['in_frames'] = frames
                inputs_map['in_fps'] = fps

                with gr.Accordion(
                    label="Flow Shift", open=False
                ):
                    flow_shift_bool = gr.Checkbox(
                        label="Enable Flow Shift", value=False
                    )
                    flow_shift = gr.Number(
                        label="Flow Shift",
                        minimum=1.0,
                        maximum=12.0,
                        value=3.0,
                        interactive=False,
                        step=0.1
                    )
                inputs_map['in_flow_shift_bool'] = flow_shift_bool
                inputs_map['in_flow_shift'] = flow_shift

                flow_shift_comp = [flow_shift]

                bottom_generation_settings_ui = create_bottom_generation_settings_ui()
                inputs_map.update(bottom_generation_settings_ui)

            with gr.Tab("Image Enhancement"):

                # Upscale
                upscl_ui = create_upscl_ui()
                inputs_map.update(upscl_ui)

                # ControlNet
                cnnet_ui = create_cnnet_ui()
                inputs_map.update(cnnet_ui)

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

                # EasyCache
                easycache_ui = create_easycache_ui()
                inputs_map.update(easycache_ui)

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
                with gr.Accordion(
                    label="Image to Video", open=False
                ):
                    img_inp_any2video = gr.Image(
                        sources="upload", type="filepath"
                    )
                    inputs_map['in_img_inp'] = img_inp_any2video
            with gr.Row():
                with gr.Accordion(
                    label="First-Last Frame Video", open=False
                ):
                    with gr.Row():
                        first_frame_inp = gr.Image(
                            sources="upload", type="filepath"
                        )
                        inputs_map['in_first_frame_inp'] = first_frame_inp
                    with gr.Row():
                        last_frame_inp = gr.Image(
                            sources="upload", type="filepath"
                        )
                        inputs_map['in_last_frame_inp'] = last_frame_inp
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
                    progress_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False,
                        visible=False,
                        label="Progress"
                    )
                with gr.Row():
                    progress_textbox = gr.Textbox(
                        label="Status:",
                        visible=False,
                        interactive=False
                    )
                with gr.Row():
                    video_final = gr.Gallery(
                        label="Generated videos",
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
                        show_copy_button=True,
                    )

    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[key] for key in ordered_keys]

    def any2video_wrapper(*args):
        """
        Accepts all UI inputs, zips them with keys, and calls the
        main any2video function.
        """
        # This line programmatically creates the dictionary.
        params = dict(zip(ordered_keys, args))
        yield from any2video(params)

    # Generate
    gen_btn.click(
        any2video_wrapper,
        inputs=ordered_components,
        outputs=[command, progress_slider, progress_textbox, stats,
                 video_final]
    )
    kill_btn.click(
        subprocess_manager.kill_subprocess,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            preview_ui['in_preview_mode'], extras_ui['in_predict']
        ]
    )

    flow_shift_bool.change(
        partial(update_interactivity, len(flow_shift_comp)),
        inputs=flow_shift_bool,
        outputs=flow_shift_comp
    )

    any2video_params['pprompt'] = prompts_ui['in_pprompt']
    any2video_params['nprompt'] = prompts_ui['in_nprompt']
    any2video_params['width'] = generation_settings_ui['in_width']
    any2video_params['height'] = generation_settings_ui['in_height']
    any2video_params['steps'] = generation_settings_ui['in_steps']
    any2video_params['sampling'] = generation_settings_ui['in_sampling']
    any2video_params['scheduler'] = generation_settings_ui['in_scheduler']
    any2video_params['cfg'] = generation_settings_ui['in_cfg']
    any2video_params['seed'] = inputs_map['in_seed']

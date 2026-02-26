"""sd.cpp-webui - Text to image UI"""

import gradio as gr

from modules.core.server.sdcpp_server import (
    start_server, stop_server, get_server_status, txt2img_api
)
from modules.utils.ui_handler import (
    ckpt_tab_switch, unet_tab_switch, refresh_all_options
)
import modules.utils.queue as queue_manager
from modules.ui.models import create_img_model_sel_ui
from modules.ui.prompts import create_prompts_ui
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
from modules.ui.chroma import create_chroma_ui
from modules.ui.qwen import create_qwen_ui
from modules.ui.circular import create_circular_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.timestep_shift import create_timestep_shift_ui
from modules.ui.eta import create_eta_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.preview import create_preview_ui
from modules.ui.performance import create_performance_ui
from modules.ui.environment import create_env_ui
# from modules.ui.experimental import create_experimental_ui


txt2img_params = {}
server_process = None


with gr.Blocks() as txt2img_server_block:
    inputs_map = {}

    # Title
    txt2img_title = gr.Markdown("# Text to Image")

    with gr.Accordion(
        label="Server configuration", open=False
    ):
        with gr.Accordion(
            label="Models selection", open=False
        ):
            # Model & VAE Selection
            model_ui = create_img_model_sel_ui()
            inputs_map.update(model_ui['inputs'])

            # Model Type Selection
            quant_ui = create_quant_ui()
            inputs_map.update(quant_ui)

        with gr.Accordion(
            label="Advanced Settings", open=False
        ):
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

            listen_ip = gr.Textbox(
                label="Listen IP",
                show_label=True,
                value="127.0.0.1",
                interactive=True
            )
            inputs_map['in_ip'] = listen_ip

            port = gr.Number(
                label="Port",
                minimum=0,
                maximum=65535,
                value=1234,
                interactive=True,
                step=1
            )
            inputs_map['in_port'] = port

    with gr.Group():
        with gr.Row():
            server_status = gr.Textbox(
                label="Server status:",
                show_label=True,
                value="Stopped",
                interactive=False
            )
            server_status_timer = gr.Timer(value=0.1, active=False)
        with gr.Row():
            server_start = gr.Button(
                value="Start server", variant="primary"
            )
            server_stop = gr.Button(
                value="Stop server", variant="stop"
            )

    # Prompts
    prompts_ui = create_prompts_ui()
    inputs_map.update(prompts_ui)

    # Settings
    with gr.Row():
        with gr.Column(scale=1):

            with gr.Tab("Generation Settings"):

                generation_settings_ui = create_generation_settings_ui()
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

                # Chroma
                chroma_ui = create_chroma_ui()
                inputs_map.update(chroma_ui)

                # Qwen
                qwen_ui = create_qwen_ui()
                inputs_map.update(qwen_ui)

                # Circular padding
                circular_ui = create_circular_ui()
                inputs_map.update(circular_ui)

                # PhotoMaker
                photomaker_ui = create_photomaker_ui()
                inputs_map.update(photomaker_ui)

                # Timestep shift
                timestep_shift_ui = create_timestep_shift_ui()
                inputs_map.update(timestep_shift_ui)

                # ETA
                eta_ui = create_eta_ui()
                inputs_map.update(eta_ui)

            # Experimental
            # experimental_ui = create_experimental_ui()
            # inputs_map.update(experimental_ui)

            with gr.Row():
                refresh_opt = gr.Button(
                    value="Refresh sd options"
                )

        # Output
        with gr.Column(scale=1):
            with gr.Group():
                with gr.Row():
                    gen_btn = gr.Button(
                        value="Generate", size="lg",
                        variant="primary"
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
                        show_reset_button=False
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
                        show_copy_button=True,
                    )

    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[key] for key in ordered_keys]

    def start_server_wrapper(*args):
        # Reconstruct the dictionary {key: value}
        params = dict(zip(ordered_keys, args))
        # Pass the dictionary to the original server function
        return start_server(params)

    server_start.click(
        fn=start_server_wrapper,
        inputs=ordered_components,
        outputs=[server_status, gen_btn, server_status_timer]
    )

    server_stop.click(
        fn=stop_server,
        inputs=[],
        outputs=[server_status, gen_btn, server_status_timer]
    )

    server_status_timer.tick(
        fn=get_server_status,
        inputs=[],
        outputs=[server_status, gen_btn]
    )

    def submit_job(*args):
        """
        Pack parameters and add to queue.
        """
        params = dict(zip(ordered_keys, args))

        # Add the API task to the queue
        queue_manager.add_job(txt2img_api, params)

        return gr.Timer(value=0.5, active=True)

    def poll_status():
        """
        Check queue status and update UI elements.
        """
        state = queue_manager.get_status()
        q_len = queue_manager.get_queue_size()

        # Decide if timer should keep running
        if state["is_running"] or q_len > 0:
            timer_update = gr.Timer(value=0.5, active=True)
        else:
            timer_update = gr.Timer(active=False)

        # Update queue text
        if q_len > 0:
            queue_update = gr.update(value=f"⏳ Jobs in queue: {q_len}", visible=True)
        else:
            queue_update = gr.update(visible=False)

        return (
            state["command"],
            state["progress"],   # slider value
            state["status"],     # status text
            state["stats"],      # stats text
            state["images"],     # gallery
            timer_update,
            queue_update
        )

    timer = gr.Timer(value=0.01, active=False)

    gen_btn.click(
        fn=submit_job,
        inputs=ordered_components,
        outputs=[timer]                      # Existing Gallery Component
    )

    timer.tick(
        poll_status,
        inputs=[],
        outputs=[
            command, progress_slider, progress_textbox,
            stats, img_final, timer, queue_tracker
        ]
    )

    # Interactive Bindings
    model_ui['components']['ckpt_tab'].select(
        ckpt_tab_switch,
        inputs=[],
        outputs=[
            model_ui['inputs']['in_diffusion_mode'],
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            model_ui['inputs']['in_llm'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ]
    )

    model_ui['components']['unet_tab'].select(
        unet_tab_switch,
        inputs=[],
        outputs=[
            model_ui['inputs']['in_diffusion_mode'],
            model_ui['inputs']['in_ckpt_model'],
            model_ui['inputs']['in_unet_model'],
            model_ui['inputs']['in_ckpt_vae'],
            model_ui['inputs']['in_unet_vae'],
            model_ui['inputs']['in_clip_g'],
            model_ui['inputs']['in_clip_l'],
            model_ui['inputs']['in_t5xxl'],
            model_ui['inputs']['in_llm'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift']
        ]
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

    txt2img_params['pprompt'] = prompts_ui['in_pprompt']
    txt2img_params['nprompt'] = prompts_ui['in_nprompt']
    txt2img_params['width'] = generation_settings_ui['in_width']
    txt2img_params['height'] = generation_settings_ui['in_height']
    txt2img_params['steps'] = generation_settings_ui['in_steps']
    txt2img_params['sampling'] = generation_settings_ui['in_sampling']
    txt2img_params['scheduler'] = generation_settings_ui['in_scheduler']
    txt2img_params['cfg'] = generation_settings_ui['in_cfg']
    txt2img_params['seed'] = inputs_map['in_seed']

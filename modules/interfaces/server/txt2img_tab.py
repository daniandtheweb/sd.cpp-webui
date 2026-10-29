"""sd.cpp-webui - Text to image UI"""

import gradio as gr

from modules.core.server.sdcpp_server import (
    txt2img_api, cancel_active_job
)
from modules.core.server.manager import (
    start_server, stop_server
)
from modules.core.server.live_status import (
    start_watcher, stop_watcher, status_stream
)
from modules.shared_instance import server_state
from modules.utils import streaming_queue as queue_manager
from modules.utils.ui_events import (
    get_ordered_inputs, bind_generation_pipeline,
    unet_tab_switch, ckpt_tab_switch,
    refresh_all_options
)
from modules.ui.models import create_img_model_sel_ui
from modules.ui.loras import (
    create_lora_sel_ui, bind_lora_events
)
from modules.ui.prompts import create_prompts_ui
from modules.ui.model_presets import (
    create_model_presets_ui, bind_model_presets_events
)
from modules.ui.generation_settings import (
    create_quant_ui, create_generation_settings_ui,
    create_bottom_generation_settings_ui
)
from modules.ui.upscale import create_upscl_ui
from modules.ui.controlnet import create_cnnet_ui
# from modules.ui.chroma import create_chroma_ui
# from modules.ui.qwen import create_qwen_ui
from modules.ui.circular import create_circular_ui
from modules.ui.photomaker import create_photomaker_ui
from modules.ui.timestep_shift import create_timestep_shift_ui
from modules.ui.eta import create_eta_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.output import create_output_ui
from modules.ui.performance import create_performance_ui
from modules.ui.deprecated import create_deprecated_ui
from modules.ui.environment import create_env_ui
# from modules.ui.experimental import create_experimental_ui


txt2img_server_params = {}
server_process = None


def _is_interactive_update(update):
    try:
        return update.get('interactive') is True
    except (AttributeError, TypeError):
        return False


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

            # Model Presets
            model_presets_ui = create_model_presets_ui()

        with gr.Accordion(
            label="Server Settings", open=False
        ):
            with gr.Tab("Image Enhancement"):

                # ControlNet
                cnnet_ui = create_cnnet_ui(inputs=False)
                cnnet_ui = {'in_cnnet': cnnet_ui['in_cnnet']}
                inputs_map.update(cnnet_ui)

                # Chroma
                # chroma_ui = create_chroma_ui()
                # inputs_map.update(chroma_ui)

                # Qwen
                # qwen_ui = create_qwen_ui()
                # inputs_map.update(qwen_ui)

                # Circular padding
                circular_ui = create_circular_ui()
                inputs_map.update(circular_ui)

                # PhotoMaker
                photomaker_ui = create_photomaker_ui()
                inputs_map.update(photomaker_ui)

            with gr.Tab("Advanced Settings"):

                # TAESD
                taesd_ui = create_taesd_ui()
                inputs_map.update(taesd_ui)

                # Extra Settings
                extras_ui = create_extras_ui(generation=False)
                del extras_ui['in_sigmas']
                inputs_map.update(extras_ui)

                # Performance Settings
                performance_ui = create_performance_ui()
                inputs_map.update(performance_ui)

                # Environment Variables
                env_ui = create_env_ui()
                inputs_map.update(env_ui)

                # Deprecated placement flags
                deprecated_ui = create_deprecated_ui()
                inputs_map.update(deprecated_ui)

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
            with gr.Column(scale=1):
                with gr.Row():
                    server_start = gr.Button(
                        value="Start server", variant="primary"
                    )
                with gr.Row():
                    server_stop = gr.Button(
                        value="Stop server", variant="stop",
                        interactive=False
                    )
            with gr.Column(scale=1):
                with gr.Row():
                    server_status = gr.Textbox(
                        label="Server & Model status:",
                        show_label=True,
                        value="Stopped (No Model Loaded)",
                        interactive=False
                    )

    # Loras
    lora_ui = create_lora_sel_ui()
    inputs_map.update(lora_ui)

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
                cnnet_inputs_ui = create_cnnet_ui(model_sel=False)
                del cnnet_inputs_ui['in_cnnet']
                inputs_map.update(cnnet_inputs_ui)

                # Timestep shift
                timestep_shift_ui = create_timestep_shift_ui()
                inputs_map.update(timestep_shift_ui)

                # ETA
                eta_ui = create_eta_ui()
                inputs_map.update(eta_ui)

            with gr.Tab("Advanced Settings"):

                # VAE Tiling
                vae_tiling_ui = create_vae_tiling_ui()
                inputs_map.update(vae_tiling_ui)

                # Cache
                cache_ui = create_cache_ui()
                inputs_map.update(cache_ui)

                # Output
                output_ui = create_output_ui()
                inputs_map.update(output_ui)

                # Extra Settings
                extras_gen_ui = create_extras_ui(launch=False)
                extras_gen_ui = {'in_sigmas': extras_gen_ui['in_sigmas']}
                inputs_map.update(extras_gen_ui)

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
                        variant="primary", interactive=False
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
                        label="Request payload:",
                        show_label=True,
                        value="",
                        interactive=False,
                        buttons=['copy'],
                    )

    ordered_keys, ordered_components = get_ordered_inputs(inputs_map)

    _server_config_keys = (
        list(model_ui['inputs'].keys())
        + list(quant_ui.keys())
        + list(cnnet_ui.keys())
        + list(circular_ui.keys())
        + list(photomaker_ui.keys())
        + list(taesd_ui.keys())
        + list(extras_ui.keys())
        + list(performance_ui.keys())
        + list(env_ui.keys())
        + list(deprecated_ui.keys())
        + ['in_ip', 'in_port']
    )
    server_config_components = [
        inputs_map[k]
        for k in _server_config_keys
        if hasattr(inputs_map.get(k), 'interactive')
    ]

    def _config_updates(is_running):
        return [
            gr.update(interactive=not is_running)
            for _ in server_config_components
        ]

    def _reset_outputs():
        return (
            gr.update(visible=False, value=0),
            gr.update(visible=False, value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=None),
            gr.update(visible=False, value=""),
        )

    def _status_stream_expanded():
        for status, btn, is_running in status_stream():
            yield (
                status,
                btn,
                gr.update(interactive=not is_running),
                gr.update(interactive=is_running),
                *_config_updates(is_running),
            )

    def start_server_wrapper(*args):
        params = dict(zip(ordered_keys, args))
        ip_val = params.get('in_ip', '127.0.0.1')
        port_val = params.get('in_port', 1234)

        prev_exit = getattr(server_state, 'last_exit_code', None)
        was_crashed = (
            not server_state.running
            and prev_exit is not None
            and prev_exit != 0
        )

        if was_crashed:
            queue_manager.clear_all()

        result = start_server(params)
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            status = result[0]
            gen_btn_update = result[2]
        else:
            status = result
            gen_btn_update = gr.skip()

        if _is_interactive_update(gen_btn_update):
            start_watcher(ip_val, port_val)

        is_running = server_state.running
        reset = _reset_outputs() if was_crashed else (gr.skip(),) * 6

        return (
            status,
            gen_btn_update,
            gr.update(interactive=not is_running),
            gr.update(interactive=is_running),
            *_config_updates(is_running),
            *reset,
        )

    def stop_server_wrapper():
        queue_manager.clear_all()

        result = stop_server()
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            status = result[0]
            gen_btn_update = result[2]
        else:
            status = result
            gen_btn_update = gr.skip()

        stop_watcher()

        is_running = server_state.running

        return (
            status,
            gen_btn_update,
            gr.update(interactive=not is_running),
            gr.update(interactive=is_running),
            *_config_updates(is_running),
            *_reset_outputs(),
        )

    _start_outputs = [
        server_status,
        gen_btn,
        server_start,
        server_stop,
        *server_config_components,
        progress_slider,
        progress_textbox,
        stats,
        command,
        img_final,
        queue_tracker,
    ]

    server_start.click(
        fn=start_server_wrapper,
        inputs=ordered_components,
        outputs=_start_outputs
    )

    server_stop.click(
        fn=stop_server_wrapper,
        inputs=[],
        outputs=_start_outputs
    )

    txt2img_server_block.load(
        fn=_status_stream_expanded,
        outputs=[
            server_status,
            gen_btn,
            server_start,
            server_stop,
            *server_config_components,
        ],
        concurrency_limit=10,
        show_progress="hidden",
    )

    bind_lora_events(lora_ui, prompts_ui)

    is_loading_preset = gr.State(value=False)

    bind_model_presets_events(
        model_presets_ui, model_ui['inputs'], generation_settings_ui,
        model_tabs=model_ui['components']['model_tabs'],
        preset_flag=is_loading_preset
    )

    ui_outputs = {
        'gen_btn': gen_btn,
        'progress_slider': progress_slider,
        'progress_textbox': progress_textbox,
        'stats': stats,
        'command': command,
        'img_final': img_final,
        'queue_tracker': queue_tracker
    }

    bind_generation_pipeline(
        txt2img_api, ordered_keys, ordered_components, ui_outputs
    )

    kill_btn.click(
        cancel_active_job,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    def safe_ckpt_tab_switch(is_loading):
        if is_loading:
            return [gr.skip()] * 15 + [False]
        return list(ckpt_tab_switch()) + [False]

    def safe_unet_tab_switch(is_loading):
        if is_loading:
            return [gr.skip()] * 15 + [False]
        return list(unet_tab_switch()) + [False]

    model_ui['components']['ckpt_tab'].select(
        safe_ckpt_tab_switch,
        inputs=[is_loading_preset],
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
            model_ui['inputs']['in_llm_vision'],
            model_ui['inputs']['in_tokenizer'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift'],
            is_loading_preset
        ]
    )

    model_ui['components']['unet_tab'].select(
        safe_unet_tab_switch,
        inputs=[is_loading_preset],
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
            model_ui['inputs']['in_llm_vision'],
            model_ui['inputs']['in_tokenizer'],
            generation_settings_ui['in_guidance_bool'],
            generation_settings_ui['in_guidance'],
            generation_settings_ui['in_flow_shift_bool'],
            generation_settings_ui['in_flow_shift'],
            is_loading_preset
        ]
    )

    refresh_opt.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            extras_ui['in_predict']
        ]
    )

    txt2img_server_params['pprompt'] = prompts_ui['in_pprompt']
    txt2img_server_params['nprompt'] = prompts_ui['in_nprompt']
    txt2img_server_params['width'] = generation_settings_ui['in_width']
    txt2img_server_params['height'] = generation_settings_ui['in_height']
    txt2img_server_params['steps'] = generation_settings_ui['in_steps']
    txt2img_server_params['sampling'] = generation_settings_ui['in_sampling']
    txt2img_server_params['scheduler'] = generation_settings_ui['in_scheduler']
    txt2img_server_params['cfg'] = generation_settings_ui['in_cfg']
    txt2img_server_params['seed'] = inputs_map['in_seed']

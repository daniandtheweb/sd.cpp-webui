"""sd.cpp-webui - Anything to Video UI (server)"""

from functools import partial

import gradio as gr

from modules.core.server.sdcpp_server import (
    any2video_api, cancel_active_job
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
    update_interactivity, refresh_all_options,
    OUTPUT_GALLERY_CLASS
)
from modules.ui.progress import create_progress_bar, progress_update
from modules.ui.models import create_video_model_sel_ui
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
from modules.ui.high_noise_generation_settings import (
    create_high_noise_generation_settings_ui
)
from modules.ui.img_preprocess import create_img_preprocess_ui
from modules.ui.slg import create_slg_ui
from modules.ui.eta import create_eta_ui
from modules.ui.taesd import create_taesd_ui
from modules.ui.vae_tiling import create_vae_tiling_ui
from modules.ui.cache import create_cache_ui
from modules.ui.extra import create_extras_ui
from modules.ui.output import create_output_ui
from modules.ui.performance import create_performance_ui
from modules.ui.deprecated import create_deprecated_ui
from modules.ui.environment import create_env_ui


any2video_server_params = {}


def _is_interactive_update(update):
    try:
        return update.get('interactive') is True
    except (AttributeError, TypeError):
        return False


with gr.Blocks() as any2video_server_block:
    inputs_map = {}

    # Title
    any2video_title = gr.Markdown("# Anything to Video")

    with gr.Accordion(
        label="Server configuration", open=False
    ):
        with gr.Accordion(
            label="Models selection", open=False
        ):
            # Model & VAE Selection
            model_ui = create_video_model_sel_ui()
            inputs_map.update(model_ui)

            # Model Type Selection
            quant_ui = create_quant_ui()
            inputs_map.update(quant_ui)

            # Model Presets
            model_presets_ui = create_model_presets_ui()

        with gr.Accordion(
            label="Server Settings", open=False
        ):
            with gr.Tab("Advanced Settings"):

                # TAESD
                taesd_ui = create_taesd_ui()
                inputs_map.update(taesd_ui)

                # Extra Settings
                extras_ui = create_extras_ui(generation=False)
                del extras_ui['in_sigmas']
                del extras_ui['in_extra_sample_args']
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

                with gr.Accordion(
                    label="Wan & MoE Specifics", open=False
                ):
                    with gr.Row():
                        moe_boundary_bool = gr.Checkbox(
                            label="Enable MoE Boundary",
                            value=False
                        )
                        moe_boundary = gr.Number(
                            label="MoE Boundary",
                            value=0.875,
                            step=0.001,
                            interactive=False
                        )
                    with gr.Row():
                        vace_strength_bool = gr.Checkbox(
                            label="Enable VACE Strength",
                            value=False
                        )
                        vace_strength = gr.Number(
                            label="VACE Strength",
                            value=1.0,
                            step=0.1,
                            interactive=False
                        )
                inputs_map['in_moe_boundary_bool'] = moe_boundary_bool
                inputs_map['in_moe_boundary'] = moe_boundary
                inputs_map['in_vace_strength_bool'] = vace_strength_bool
                inputs_map['in_vace_strength'] = vace_strength

                bottom_generation_settings_ui = (
                    create_bottom_generation_settings_ui()
                )
                # The video request has no batch count
                bottom_generation_settings_ui['in_batch_count'].visible = False
                del bottom_generation_settings_ui['in_batch_count']
                inputs_map.update(bottom_generation_settings_ui)

            with gr.Tab("High Noise Settings"):
                high_noise_generation_settings_ui = (
                    create_high_noise_generation_settings_ui()
                )
                inputs_map.update(high_noise_generation_settings_ui)

            with gr.Tab("Image Enhancement"):

                # Image Preprocessing
                img_preprocess_ui = create_img_preprocess_ui()
                inputs_map.update(img_preprocess_ui)

                # Skip Layer Guidance
                slg_ui = create_slg_ui()
                inputs_map.update(slg_ui)

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
                # The server names and numbers the files itself
                for key in ('in_output', 'in_output_begin_idx',
                            'in_disable_img_metadata'):
                    output_ui[key].visible = False
                    del output_ui[key]
                inputs_map.update(output_ui)

                # Extra Settings
                extras_gen_ui = create_extras_ui(launch=False)
                extras_gen_ui = {
                    k: extras_gen_ui[k]
                    for k in ('in_extra_sample_args',)
                }
                inputs_map.update(extras_gen_ui)

            with gr.Row():
                refresh_choices = gr.Button(
                    value="Refresh sd choices"
                )

        # Output
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Accordion(
                    label="Image to Video", open=False
                ):
                    img_inp_any2video_server = gr.Image(
                        sources="upload", type="filepath"
                    )
                    inputs_map['in_img_inp'] = img_inp_any2video_server
            with gr.Row():
                with gr.Accordion(
                    label="Speech to Video (S2V)", open=False
                ):
                    gr.Markdown(
                        "Wan2.2 S2V: the portrait goes in Image to Video "
                        "and needs the audio_encoder model and the "
                        "wan_2.1 VAE. Runs at 16 FPS, pick about "
                        "16 frames per second of audio (up to 81)."
                    )
                    s2v_audio = gr.Audio(
                        label="Driving audio",
                        sources=["upload"],
                        type="filepath",
                        format="wav"
                    )
                    inputs_map['in_s2v_audio'] = s2v_audio
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
            with gr.Row():
                with gr.Accordion(
                    label="Video to Video (VACE)", open=False
                ):
                    control_video_dir = gr.Textbox(
                        label="Control Video Frames Directory",
                        placeholder=(
                            "Path to folder containing extracted frames "
                            "(e.g., ./post+depth)"
                        )
                    )
                    inputs_map['in_control_video_dir'] = control_video_dir
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
                    progress_bar = create_progress_bar()
                with gr.Row():
                    video_final = gr.Gallery(
                        label="Generated videos",
                        show_label=False,
                        columns=[3],
                        rows=[1],
                        object_fit="contain",
                        height="auto",
                        buttons=["share", "download", "fullscreen"],
                        elem_classes=[OUTPUT_GALLERY_CLASS]
                    )
                with gr.Row():
                    stats = gr.Textbox(
                        label="Statistics:",
                        show_label=True,
                        value="",
                        interactive=False,
                        visible=False
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
        list(model_ui.keys())
        + list(quant_ui.keys())
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
        # Keep the command, clear the rest
        return (
            progress_update(visible=False),
            gr.update(visible=False, value=""),
            gr.skip(),
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
            start_watcher(server_state.ip, server_state.port)

        is_running = server_state.running
        reset = _reset_outputs() if was_crashed else (gr.skip(),) * 5

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
        progress_bar,
        stats,
        command,
        video_final,
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

    any2video_server_block.load(
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
        model_presets_ui, model_ui, generation_settings_ui,
        preset_flag=is_loading_preset
    )

    ui_outputs = {
        'gen_btn': gen_btn,
        'progress_bar': progress_bar,
        'stats': stats,
        'command': command,
        'img_final': video_final,
        'queue_tracker': queue_tracker
    }

    bind_generation_pipeline(
        any2video_api, ordered_keys, ordered_components, ui_outputs
    )

    kill_btn.click(
        cancel_active_job,
        inputs=[],
        outputs=[]
    )

    # Interactive Bindings
    refresh_choices.click(
        refresh_all_options,
        inputs=[],
        outputs=[
            generation_settings_ui['in_sampling'],
            generation_settings_ui['in_scheduler'],
            extras_ui['in_predict']
        ]
    )

    flow_shift_bool.change(
        partial(update_interactivity, len(flow_shift_comp)),
        inputs=flow_shift_bool,
        outputs=flow_shift_comp
    )

    moe_boundary_bool.change(
        partial(update_interactivity, 1),
        inputs=moe_boundary_bool,
        outputs=moe_boundary
    )

    vace_strength_bool.change(
        partial(update_interactivity, 1),
        inputs=vace_strength_bool,
        outputs=vace_strength
    )

    any2video_server_params['pprompt'] = prompts_ui['in_pprompt']
    any2video_server_params['nprompt'] = prompts_ui['in_nprompt']
    any2video_server_params['width'] = generation_settings_ui['in_width']
    any2video_server_params['height'] = generation_settings_ui['in_height']
    any2video_server_params['steps'] = generation_settings_ui['in_steps']
    any2video_server_params['sampling'] = generation_settings_ui['in_sampling']
    any2video_server_params['scheduler'] = (
        generation_settings_ui['in_scheduler']
    )
    any2video_server_params['cfg'] = generation_settings_ui['in_cfg']
    any2video_server_params['seed'] = inputs_map['in_seed']

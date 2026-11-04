"""sd.cpp-webui - server tabs - the server controls the tabs share"""

import gradio as gr

from modules.core.server.manager import (
    start_server, stop_server
)
from modules.core.server.live_status import (
    start_watcher, stop_watcher, status_stream
)
from modules.shared_instance import server_state
from modules.utils import streaming_queue as queue_manager
from modules.ui.progress import progress_update


def _is_interactive_update(update):
    try:
        return update.get('interactive') is True
    except (AttributeError, TypeError):
        return False


def create_listen_ui():
    """Create the address the server listens on"""
    listen_ip = gr.Textbox(
        label="Listen IP",
        show_label=True,
        value="127.0.0.1",
        interactive=True
    )

    port = gr.Number(
        label="Port",
        minimum=0,
        maximum=65535,
        value=1234,
        interactive=True,
        step=1
    )

    return {'in_ip': listen_ip, 'in_port': port}


def create_server_controls():
    """Create the start and stop buttons and the server status"""
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

    return {
        'start': server_start,
        'stop': server_stop,
        'status': server_status
    }


def bind_server_controls(
    block, controls, gen_btn, outputs, config_keys,
    inputs_map, ordered_keys, ordered_components
):
    """
    Bind the server controls of a tab. The outputs are its progress bar,
    statistics, command, gallery and queue tracker; the config keys are the
    settings locked while the server is running.
    """
    server_start = controls['start']
    server_stop = controls['stop']
    server_status = controls['status']

    server_config_components = [
        inputs_map[k]
        for k in config_keys
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
        *outputs,
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

    block.load(
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

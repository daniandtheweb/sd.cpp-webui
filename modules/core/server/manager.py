"""sd.cpp-webui - core - stable-diffusion.cpp server manager"""

import threading
from typing import Dict, Any

import gradio as gr

from modules.core.common.sd_common import (
    CommonRunner
)
from modules.shared_instance import (
    subprocess_manager, SD_SERVER, server_state
)
from modules.options import convert, emit


class ServerRunner(CommonRunner):
    """
    Builds and manages the sd-server command execution.
    """

    kind = 'server'

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # The video tab starts the server with its own models
        if 'in_video_mode' in params:
            self.kind = 'video'
        self.command = [SD_SERVER]

    def build_command(self):
        """
        Constructs the server arguments. The options sent per request
        (those with a payload field in the registry) are not part of it.
        """
        self._resolve_paths()

        self.ip = convert.listen_ip(self._get_param('in_ip'), None)
        self.port = convert.listen_port(self._get_param('in_port'), None)

        self.command.extend(emit.launch(self._context()))

    def _prepare_for_run(self):
        """Prepares the final command string for printing."""
        self.fcommand = ' '.join(map(str, self.command))

    def run(self):
        """Starts the server thread and handles initial setup/logging."""
        self._prepare_for_run()
        print(f"\n\n{self.fcommand}\n\n")

        process_env = self._build_process_env()

        def run_server_wrapper():
            server_state.running = True
            server_state.last_exit_code = None
            server_state.user_stopped = False

            try:
                final_stats_str = "Process completed with unknown stats."
                for update in subprocess_manager.run_subprocess(
                    self.command, env=process_env
                ):
                    if "final_stats" in update:
                        stats = update["final_stats"]
                        final_stats_str = (
                            f"Sampling: {stats.get('sampling_time', 'N/A')} | "
                            f"Decode: {stats.get('decoding_time', 'N/A')} | "
                            f"Total: {stats.get('total_time', 'N/A')} | "
                            f"Last Speed: {stats.get('last_speed', 'N/A')}"
                        )
                        server_state.last_generation_stats = final_stats_str

                        server_state.latest_update = {"status": final_stats_str}
                    else:
                        server_state.latest_update = update
            except Exception as e:
                print(f"[SD-Server] Crashed: {e}")
                server_state.last_exit_code = -1
            finally:
                if server_state.last_exit_code is None:
                    if server_state.user_stopped:
                        server_state.last_exit_code = 0
                    else:
                        server_state.last_exit_code = -1
                server_state.user_stopped = False
                server_state.running = False

        thread = threading.Thread(target=run_server_wrapper, daemon=True)
        thread.start()

        # Address of the running server, shared by every tab
        server_state.ip = self.ip
        server_state.port = self.port
        server_state.running = True
        return "Running", gr.update(active=True), gr.update(interactive=True)


def start_server(params):
    """Start the sd-server subprocess with validated paths."""

    if server_state.running:
        return "Running", gr.update(active=True), gr.update(interactive=True)

    try:
        runner = ServerRunner(params)
        runner.build_command()
        return runner.run()

    except Exception as e:
        return f"Error: {e}", gr.update(active=False), gr.update(interactive=False)


def stop_server():
    """Stop the sd-server subprocess."""
    if not server_state.running:
        return "Stopped", gr.update(active=False), gr.update(interactive=False)

    try:
        server_state.user_stopped = True
        server_state.running = False
        subprocess_manager.kill_subprocess()

        return (
            "Stopped",
            gr.update(active=False),
            gr.update(interactive=False)
        )

    except Exception:
        server_state.user_stopped = False
        server_state.running = True
        return (
            "Error",
            gr.update(active=False),
            gr.update(interactive=False)
        )

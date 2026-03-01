"""sd.cpp-webui - core - stable-diffusion.cpp server manager"""

import threading
from typing import Dict, Any

import gradio as gr

from modules.core.common.sd_common import (
    CommonRunner
)
from modules.shared_instance import (
    config, subprocess_manager, SD_SERVER, server_state
)


class ServerRunner(CommonRunner):
    """
    Builds and manages the sd-server command execution.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.command = [SD_SERVER]

    def build_command(self):
        """Constructs the server arguments."""
        self._resolve_paths()

        # Network settings
        self.command.extend([
            "--listen-ip", str(self._get_param('ip', '127.0.0.1')),
            "--listen-port", str(self._get_param('port', 1234)),
        ])

        options = self._get_common_model_options()

        # Additional Components
        options.update({
            '--threads': self._get_param('in_threads'),
            '--taesd': self._get_param('f_taesd'),
            '--photo-maker': self._get_param('f_phtmkr'),
            '--upscale-model': self._get_param('f_upscl'),
            '--control-net': self._get_param('f_cnnet'),
            '--embd-dir': config.get('emb_dir'),
            '--lora-model-dir': config.get('lora_dir'),
        })

        self._add_options(options)

        self._add_flags(self._get_common_flags())

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

                        server_state.latest_update = {"status": final_stats_str, "percent": 100}
                    else:
                        server_state.latest_update = update
            except Exception as e:
                print(f"[SD-Server] Crashed: {e}")
            finally:
                server_state.running = False

        thread = threading.Thread(target=run_server_wrapper, daemon=True)
        thread.start()

        server_state.running = True
        return "Running", gr.update(interactive=True)


def start_server(params):
    """Start the sd-server subprocess with validated paths."""

    if server_state.running:
        return "Running", gr.skip()

    try:
        runner = ServerRunner(params)
        runner.build_command()
        return runner.run()

    except Exception as e:
        return f"Error: {e}", gr.update(interactive=False)


def stop_server():
    """Stop the sd-server subprocess."""
    if not server_state.running:
        return "Stopped", gr.update(interactive=False)

    try:
        # Use the manager to kill the process
        subprocess_manager.kill_subprocess()
        server_state.running = False

        return "Stopped", gr.update(interactive=False)

    except Exception:
        return "Error", gr.update(interactive=False)

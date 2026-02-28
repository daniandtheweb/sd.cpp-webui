"""sd.cpp-webui - core - stable-diffusion.cpp server manager"""

import os
import threading
from enum import IntEnum
from typing import Dict, Any

import gradio as gr

from modules.utils.utility import get_path
from modules.utils.sdcpp_utils import extract_env_vars
from modules.shared_instance import (
    config, subprocess_manager, SD_SERVER, server_state
)
from modules.ui.constants import CIRCULAR_PADDING


class DiffusionMode(IntEnum):
    CHECKPOINT = 0
    UNET = 1


class ServerRunner:
    """
    Builds and manages the sd-server command execution.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_vars = extract_env_vars(self.params)
        self.command = [SD_SERVER]
        self.fcommand = ""

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from the params dictionary."""
        return self.params.get(key, default)

    def _resolve_paths(self):
        """Resolves all model and directory paths from the config."""
        path_mappings = {
            'ckpt_dir': ['in_ckpt_model'],
            'vae_dir': ['in_ckpt_vae', 'in_unet_vae'],
            'unet_dir': ['in_unet_model', 'in_high_noise_model'],
            'txt_enc_dir': [
                'in_clip_g', 'in_clip_l', 'in_t5xxl', 'in_llm',
                'in_umt5_xxl', 'in_clip_vision_h'
            ],
            'taesd_dir': ['in_taesd'],
            'phtmkr_dir': ['in_phtmkr'],
            'upscl_dir': ['in_upscl'],
            'cnnet_dir': ['in_cnnet']
        }
        for dir_key, param_keys in path_mappings.items():
            for param_key in param_keys:
                if param_key in self.params:
                    # Create a new key for the full path, e.g., 'f_ckpt_model'
                    full_path_key = f"f_{param_key.replace('in_', '')}"
                    self.params[full_path_key] = get_path(
                        config.get(dir_key), self.params.get(param_key)
                    )

    def _add_options(self, options: Dict[str, Any]):
        """Adds key-value options to the command if the value is not None."""
        for opt, val in options.items():
            if val is not None:
                self.command.extend([opt, str(val)])

    def _add_flags(self, flags: Dict[str, bool]):
        """Adds boolean flags to the command if they are True."""
        for flag, condition in flags.items():
            if condition:
                self.command.append(flag)

    def build_command(self):
        """Constructs the server arguments."""
        self._resolve_paths()

        # Network settings
        self.command.extend([
            "--listen-ip", str(self._get_param('ip', '127.0.0.1')),
            "--listen-port", str(self._get_param('port', 1234)),
        ])

        # Core logic depending on mode
        diffusion_mode = self._get_param('in_diffusion_mode')

        # Base model options
        options = {}

        if diffusion_mode == DiffusionMode.CHECKPOINT:
            options['--model'] = self._get_param('f_ckpt_model')
            options['--vae'] = self._get_param('f_ckpt_vae')
        elif diffusion_mode == DiffusionMode.UNET:
            options['--diffusion-model'] = self._get_param('f_unet_model')
            options['--vae'] = self._get_param('f_unet_vae')
            options['--clip_g'] = self._get_param('f_clip_g')
            options['--clip_l'] = self._get_param('f_clip_l')
            options['--t5xxl'] = self._get_param('f_t5xxl')
            options['--llm'] = self._get_param('f_llm')
            options['--llm_vision'] = self._get_param('f_llm_vision')

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

        # Boolean Flags
        flags = {
            '--offload-to-cpu': self._get_param('in_offload_to_cpu'),
            '--vae-tiling': self._get_param('in_vae_tiling'),
            '--vae-on-cpu': self._get_param('in_vae_cpu'),
            '--clip-on-cpu': self._get_param('in_clip_cpu'),
            '--control-net-cpu': self._get_param('in_cnnet_cpu'),
            '--canny': self._get_param('in_canny'),
            '--chroma-disable-dit-mask': (
                self._get_param('in_disable_dit_mask')
            ),
            '--chroma-enable-t5-mask': self._get_param('in_enable_t5_mask'),
            '--qwen-image-zero-cond-t': (
                self._get_param('in_enable_zero_cond_t')
            ),
            '--circular': (
                self._get_param('in_circular_padding') == CIRCULAR_PADDING[1]
            ),
            '--circularx': (
                self._get_param('in_circular_padding') == CIRCULAR_PADDING[2]
            ),
            '--circulary': (
                self._get_param('in_circular_padding') == CIRCULAR_PADDING[3]
            ),
            '--fa': self._get_param('in_flash_attn'),
            '--diffusion-fa': self._get_param('in_diffusion_fa'),
            '--diffusion-conv-direct': (
                self._get_param('in_diffusion_conv_direct')
            ),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '--force-sdxl-vae-conv-scale': (
                self._get_param('in_force_sdxl_vae_conv_scale')
            ),
            '--mmap': self._get_param('in_mmap'),
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose')
        }
        self._add_flags(flags)

    def _prepare_for_run(self):
        """Prepares the final command string for printing."""
        self.fcommand = ' '.join(map(str, self.command))

    def run(self):
        """Starts the server thread and handles initial setup/logging."""
        self._prepare_for_run()
        print(f"\n\n{self.fcommand}\n\n")

        process_env = os.environ.copy()

        if self.env_vars:
            settings_to_print = []

            for key, value in self.env_vars.items():
                if isinstance(value, bool):
                    if value is True:
                        process_env[key] = "1"
                        settings_to_print.append(f"{key}=1")
                elif isinstance(value, int):
                    process_env[key] = str(value)
                    settings_to_print.append(f"{key}={str(value)}")

            if settings_to_print:
                full_line = " ".join(settings_to_print)
                print(f"  SET: {full_line}\n\n")

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
                        print(f"[SD-Server] Session Stats: {final_stats_str}")

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

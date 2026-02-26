"""sd.cpp-webui - core - stable-diffusion.cpp server"""

import os
import io
import json
import base64
import requests
import threading
from PIL import Image
from enum import IntEnum
from typing import Dict, Any

import gradio as gr

from modules.utils.utility import get_path
from modules.utils.sdcpp_utils import (
    extract_env_vars, generate_output_filename
)
from modules.shared_instance import config, subprocess_manager, SD_SERVER

server_running = False


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
            '--fa': True,  # Defaulting to True based on original code, or make param
            '--vae-conv-direct': True,
            '-v': self._get_param('in_verbose', False)
        }
        self._add_flags(flags)

    def start(self):
        """Starts the server thread."""
        global server_running

        cmd_str = " ".join(self.command)
        print(f"[SD-Server] Starting: {cmd_str}")

        def run_server_wrapper():
            global server_running
            server_running = True

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
                    print(f"[SD-Server] SET: {full_line}")

            try:
                # Iterate over generator to keep process alive and log output
                for _ in subprocess_manager.run_subprocess(self.command, env=process_env):
                    pass
            except Exception as e:
                print(f"[SD-Server] Crashed: {e}")
            finally:
                server_running = False
                print("[SD-Server] Terminated.")

        thread = threading.Thread(target=run_server_wrapper, daemon=True)
        thread.start()

        server_running = True
        return "Running", gr.update(interactive=True)


def start_server(params):
    """Start the sd-server subprocess with validated paths."""
    global server_running

    if server_running:
        return "Running", gr.update(interactive=False)

    try:
        # Extract necessary params for path validation checks if needed here,
        # or rely on the ServerRunner to fail if paths are missing.
        # Ideally, we map the UI inputs to the standard keys expected by Runner.

        # Note: Ensure the 'params' passed in has keys matching
        # 'in_diffusion_mode', 'in_ckpt_model', etc.

        runner = ServerRunner(params)
        runner.build_command()
        return runner.start()

    except Exception as e:
        return f"Error: {e}", gr.update(interactive=False)


def stop_server():
    """Stop the sd-server subprocess."""
    global server_running

    if not server_running:
        return "Stopped", gr.update(interactive=False)

    try:
        # Use the manager to kill the process
        subprocess_manager.kill_subprocess()
        server_running = False

        return "Stopped", gr.update(interactive=False)

    except Exception:
        return "Error", gr.update(interactive=False)


def get_server_status():
    """
    Check if the server is actually running.
    Bind this to a gr.Timer or poll it to update the UI if the server crashes.
    """
    global server_running
    if server_running:
        return "Running", gr.update(interactive=True)
    return "Stopped", gr.update(interactive=False)


def api_generation_task(params):
    """
    Generator function compatible with queue.py.
    Yields: [command, progress, status, stats, images]
    """
    ip = params.get('ip', '127.0.0.1')
    port = params.get('port', 1234)
    prompt = params.get('prompt', '')
    width = params.get('width', 512)
    height = params.get('height', 512)

    url = f"http://{ip}:{int(port)}/v1/images/generations"
    headers = {"Content-Type": "application/json"}

    yield (f"POST {url}", 0, "Connecting to server...", "Waiting...", None)

    payload = {
        "model": "default",
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
        "response_format": "b64_json"
    }

    try:
        # 2. Yield Sending State
        yield (f"POST {url}", 20, "Sending request...", "Processing...", None)

        # Blocking call to the server
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=None
        )

        if response.status_code == 200:
            data = response.json()
            images = []

            for item in data.get('data', []):
                b64_data = item['b64_json']
                image = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                images.append(image)

            yield (f"Success: {url}", 100, "Done", f"Generated {len(images)} image(s)", images)

        else:
            error_msg = f"Server Error {response.status_code}: {response.text}"
            yield (f"Failed: {url}", 0, "Error", error_msg, None)

    except Exception as e:
        yield ("Connection Failed", 0, "Error", str(e), None)

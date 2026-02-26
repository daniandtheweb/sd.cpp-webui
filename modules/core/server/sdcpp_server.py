"""sd.cpp-webui - core - stable-diffusion.cpp server"""

import os
import io
import json
import base64
import requests
import threading
from PIL import Image
from enum import IntEnum
from typing import Dict, Any, Generator

import gradio as gr

from modules.utils.utility import get_path
from modules.utils.sdcpp_utils import (
    extract_env_vars, generate_output_filename
)
from modules.shared_instance import config, subprocess_manager, SD_SERVER
from modules.ui.constants import CIRCULAR_PADDING

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
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose')
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
        return "Running", gr.update(interactive=True), gr.update(active=True)


def start_server(params):
    """Start the sd-server subprocess with validated paths."""
    global server_running

    if server_running:
        return "Running", gr.update(), gr.update()

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
        return f"Error: {e}", gr.update(interactive=False), gr.update()


def stop_server():
    """Stop the sd-server subprocess."""
    global server_running

    if not server_running:
        return "Stopped", gr.update(interactive=False), gr.update()

    try:
        # Use the manager to kill the process
        subprocess_manager.kill_subprocess()
        server_running = False

        return "Stopped", gr.update(interactive=False), gr.update(active=False)

    except Exception:
        return "Error", gr.update(interactive=False), gr.update(active=False)


def get_server_status():
    """
    Check if the server is actually running.
    """
    global server_running
    if server_running:
        return "Running", gr.update(interactive=True)
    return "Stopped", gr.update(interactive=False)


class ApiTaskRunner:
    """
    Builds and manages API requests to the sd.cpp server.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.ip = str(self._get_param('in_ip'))
        self.port = str(self._get_param('in_port'))
        self.url = f"http://{self.ip}:{self.port}/v1/images/generations"

        self.output_path = ""
        self.outputs = []
        self.fcommand = f"POST {self.url}"

    def _set_output_path(self, dir_key: str, subctrl_id: int, extension: str):
        """Determines and sets the output path for the command."""
        output_dir = config.get(dir_key)
        filename_override = self._get_param('in_output')
        output_scheme = config.get('def_output_scheme')

        if filename_override and str(filename_override).strip():
            filename = f"{filename_override}.{extension}"
            self.output_path = os.path.join(output_dir, filename)
            return

        name_parts = []

        if config.get('def_output_steps'):
            steps_val = self._get_param('in_steps')
            if steps_val:
                name_parts.append(f"{steps_val}_steps")

        if config.get('def_output_quant'):
            quant_val = self._get_param('in_model_type')
            if quant_val and quant_val != "Default":
                name_parts.append(str(quant_val))

        self.output_path = generate_output_filename(
            output_dir, output_scheme, extension,
            name_parts, subctrl_id
        )

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

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from the params dictionary."""
        return self.params.get(key, default)

    def _build_payload(self) -> dict:
        """Constructs the JSON payload from UI parameters."""
        payload = {
            "model": "default",
            "prompt": self._get_param('in_pprompt', ''),
            "n": self._get_param('in_batch_count', 1),
            "size": f"{self._get_param('in_width')}x{self._get_param('in_height')}",
            "response_format": "b64_json"
        }

        # Format: 'ui_key': ('api_key', type, 'bool_condition_key' or None)
        mapping = {
            'in_nprompt':   ('negative_prompt', str, None),
            'in_sampling':  ('sample_method', str, None),
            'in_scheduler': ('scheduler', str, None),
            'in_steps':     ('steps', int, None),
            'in_cfg':       ('cfg_scale', float, None),
            'in_flow_shift': ('flow_shift', float, 'in_flow_shift_bool'),
            'in_guidance': ('guidance', float, 'in_guidance_bool'),
            'in_seed':      ('seed', int, None),
            'in_clip_skip': ('clip_skip', int, None),
        }

        extra_args = {}
        for p_key, (a_key, cast_type, cond_key) in mapping.items():
            if cond_key and not self._get_param(cond_key, False):
                continue

            val = self._get_param(p_key)

            if val is not None and str(val).strip() != "":
                try:
                    extra_args[a_key] = cast_type(val)
                except (ValueError, TypeError):
                    continue

        if extra_args:
            payload["prompt"] += f"<sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>"

        return payload

    def _process_response(self, data: dict):
        """Decodes and saves images, populating self.outputs."""
        base, ext = os.path.splitext(self.output_path)

        for i, item in enumerate(data.get("data", [])):
            if b64_data := item.get("b64_json"):
                image = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                target_path = self.output_path if i == 0 else f"{base}_{i + 1}{ext}"
                image.save(target_path)
                self.outputs.append(target_path)

    def run(self) -> Generator:
        """Generator yielding updates identical to CLI runners."""
        self._resolve_paths()

        payload = self._build_payload()

        yield (self.fcommand, gr.update(visible=True, value=10),
               "Connecting...", "Sending Request", None)

        try:
            response = requests.post(self.url, json=payload, timeout=None)

            if response.status_code == 200:
                yield (self.fcommand, gr.update(value=80), "Decoding...", "Processing Response", None)
                self._process_response(response.json())

                yield (self.fcommand, gr.update(visible=False, value=100),
                       "Done", f"Saved to {os.path.dirname(self.output_path)}", self.outputs)
            else:
                yield (self.fcommand, gr.update(value=0), "Error", f"Status {response.status_code}", None)

        except Exception as e:
            yield (self.fcommand, gr.update(value=0), "Connection Failed", str(e), None)


class Txt2ImgApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(dir_key='txt2img_dir', subctrl_id=0, extension='png')


def txt2img_api(params: dict) -> Generator:
    """Creates and runs a Txt2ImgApiRunner."""
    runner = Txt2ImgApiRunner(params)
    runner.prepare()
    yield from runner.run()

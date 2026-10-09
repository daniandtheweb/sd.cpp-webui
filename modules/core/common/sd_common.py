"""sd.cpp-webui - core - stable-diffusion.cpp common"""

import math
import os
import re
from PIL import Image
from enum import IntEnum
from typing import Dict, Any

from modules.utils.file_utils import get_path
from modules.utils.sdcpp_utils import extract_env_vars, generate_output_filename
from modules.shared_instance import config, SD_CLI
from modules.ui.constants import CIRCULAR_PADDING

LORA_TAG_PATTERN = re.compile(r'<lora:([^:]+):([^>]+)>')


def image_to_pil(value):
    """Coerces a gradio image value into a PIL Image."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        value = value[0]
    if isinstance(value, dict):
        value = value.get('path') or value.get('name')
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return Image.open(value)
    return Image.fromarray(value)


def process_editor_mask(mask_input: Any) -> Image.Image | None:
    """
    Parses the Gradio ImageEditor input.
    Returns a PIL Image: a generated white-on-black mask if drawn,
    or the uploaded pre-made mask if the drawing layer is empty.
    """
    if not mask_input:
        return None

    if isinstance(mask_input, dict):
        background_path = mask_input.get("background")
        layers = mask_input.get("layers", [])

        if layers and layers[0]:
            layer_path = layers[0]
            try:
                layer_img = Image.open(layer_path).convert("RGBA")
                # Check if the user actually drew anything (alpha channel > 0)
                max_alpha = layer_img.getextrema()[3][1]

                if max_alpha > 0:
                    mask_img = Image.new("RGB", layer_img.size, "black")
                    white_fill = Image.new("RGB", layer_img.size, "white")
                    mask_img.paste(white_fill, mask=layer_img.split()[3])
                    return mask_img

                # No drawing strokes, fallback to background
                if background_path:
                    return Image.open(background_path)
            except Exception as e:
                print(f"Error processing mask layer: {e}")
                if background_path:
                    return Image.open(background_path)
        elif background_path:
            return Image.open(background_path)

    return None


class DiffusionMode(IntEnum):
    CHECKPOINT = 0
    UNET = 1


class CommonRunner():
    """
    Common class containing shared logic for CLI and server runners.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.env_vars = extract_env_vars(self.params)
        self.command = []
        self.fcommand = ""

    def _get_param(self, key: str, default: Any = None) -> Any:
        """
        Helper to get a parameter from the params dictionary.
        """
        return self.params.get(key, default)

    # Canonical module names and the aliases the CLI accepts for them
    _BACKEND_MODULES = {
        'diffusion': {'diffusion', 'model', 'unet', 'dit'},
        'te': {'te', 'clip', 'text', 'textencoder', 'textencoders',
               'conditioner', 'cond', 'llm', 't5', 't5xxl'},
        'clip_vision': {'clip_vision', 'clipvision', 'vision'},
        'vae': {'vae', 'firststage', 'autoencoder', 'tae'},
        'controlnet': {'controlnet', 'control'},
        'photomaker': {'photomaker', 'photomakerid', 'pmid', 'photo'},
        'upscaler': {'upscaler', 'esrgan', 'hires'},
        'detector': {'detector', 'adetailer', 'yolo'},
        'audio_encoder': {'audio_encoder', 'audioencoder', 'audio'},
    }
    _BACKEND_ALIAS_MAP = {
        alias.replace('-', '').replace('_', ''): module
        for module, aliases in _BACKEND_MODULES.items()
        for alias in aliases
    }
    _DEFAULT_ALIASES = {'*', 'primary', 'all', 'default'}

    @staticmethod
    def _norm_name(name: str) -> str:
        return str(name).strip().lower().replace('-', '').replace('_', '')

    def _parse_backend_table(self, param_key: str = 'in_backend_table',
                             params_backend: bool = False) -> str | None:
        """
        Parses the backend table rows into a valid sd.cpp backend string.
        Example output: "vulkan0,diffusion=vulkan0&vulkan1,vae=cpu"
        """
        backend_table = self._get_param(param_key)

        # If UI didn't pass it or it's empty, return None
        if not backend_table or not isinstance(backend_table, list):
            return None

        parts = []
        default_device = None

        for row in backend_table:
            if len(row) < 2:
                continue
            component = self._norm_name(row[0])
            device = str(row[1]).strip().lower()

            # Skip invalid/default rows
            if not device or device in ('default', 'auto', ''):
                continue
            # 'disk' is a parameter residency mode, not a compute backend
            if not params_backend and device == 'disk':
                continue

            if component in self._DEFAULT_ALIASES:
                default_device = device
                continue

            module = self._BACKEND_ALIAS_MAP.get(component)
            if not module:
                continue
            parts.append(f"{module}={device}")

        # Default entry first, per-module assignments override it
        if default_device:
            entry = (f"*={default_device}"
                     if params_backend else default_device)
            parts.insert(0, entry)

        return ",".join(parts) if parts else None

    # Only these modules support layer/row splitting of their compute
    _SPLIT_MODULES = {'diffusion', 'te'}

    def _parse_split_modes(self) -> str | None:
        """
        Parses the per-module split mode rows into a '--split-mode'
        value. Example output: "diffusion=row,te=layer"
        """
        rows = self._get_param('in_split_modes')
        if not rows or not isinstance(rows, list):
            return None

        parts = []
        for row in rows:
            if not row or len(row) < 2:
                continue
            module = str(row[0]).strip().lower()
            mode = str(row[1]).strip().lower()
            if module in self._SPLIT_MODULES and mode in ('layer', 'row'):
                parts.append(f"{module}={mode}")
        return ",".join(parts) if parts else None

    # Budget keys the CLI accepts in place of a device name
    _MAX_VRAM_DEFAULT_KEYS = {'', 'default', 'all', '*'}

    def _parse_max_vram_table(self) -> str | None:
        """
        Parses the max-vram table rows into a valid sd.cpp budget string.
        Example output: "cuda0=6,vulkan0=2"
        """
        rows = self._get_param('in_max_vram_table')
        if not rows or not isinstance(rows, list):
            return None

        parts = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            device = str(row[0] or '').strip()
            value = str(row[1] or '').strip()
            if not value or device.lower() == 'disk':
                continue
            try:
                budget = float(value)
            except ValueError:
                continue
            if not math.isfinite(budget):
                continue
            if device.lower() in self._MAX_VRAM_DEFAULT_KEYS:
                parts.append(value)
            else:
                parts.append(f"{device}={value}")
        return ",".join(parts) if parts else None

    def _make_relative(self, path):
        """Converts paths inside the exe directory to relative form."""
        if not path or not os.path.isabs(str(path)):
            return path
        try:
            exe_dir = os.path.dirname(os.path.abspath(SD_CLI))
            rel = os.path.relpath(str(path), start=exe_dir)
            # Paths outside the exe directory stay absolute
            if not rel.startswith('..'):
                return rel
        except ValueError:
            # Fallback for cross-drive paths on Windows
            pass
        return path

    def _set_output_path(self, dir_key: str, subctrl_id: int, extension: str):
        """Determines and sets the output path for the command."""
        output_dir = config.get(dir_key)
        filename_override = self._get_param('in_output')
        output_scheme = config.get('def_output_scheme')

        if filename_override and str(filename_override).strip():
            base_name = str(filename_override).strip()
            filename = f"{base_name}.{extension}"
            test_path = os.path.join(output_dir, filename)

            counter = 1
            while os.path.exists(test_path):
                filename = f"{base_name}_{counter}.{extension}"
                test_path = os.path.join(output_dir, filename)
                counter += 1

            self.output_path = self._make_relative(test_path)
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

        self.output_path = self._make_relative(generate_output_filename(
            output_dir, output_scheme, extension,
            name_parts, subctrl_id
        ))

    def _resolve_paths(self):
        """
        Resolves all model and directory paths from the config.
        """
        path_mappings = {
            'ckpt_dir': ['in_ckpt_model'],
            'vae_dir': ['in_ckpt_vae', 'in_unet_vae', 'in_audio_vae'],
            'unet_dir': ['in_unet_model', 'in_high_noise_model', 'in_uncond_unet_model'],
            'txt_enc_dir': [
                'in_clip_g', 'in_clip_l', 'in_t5xxl', 'in_llm',
                'in_llm_vision', 'in_umt5_xxl', 'in_clip_vision_h',
                'in_emb_connect'
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
        """
        Adds key-value options to the command if the value is not None.
        """
        for opt, val in options.items():
            if val is not None:
                self.command.extend([opt, str(val)])

    def _add_flags(self, flags: Dict[str, bool]):
        """Adds boolean flags to the command if they are True."""
        for flag, condition in flags.items():
            if condition:
                self.command.append(flag)

    def _get_common_model_options(self) -> Dict[str, Any]:
        """
        Returns the base model options.
        """
        options = {}
        diffusion_mode = self._get_param('in_diffusion_mode')

        if diffusion_mode == DiffusionMode.CHECKPOINT:
            options['--model'] = self._make_relative(self._get_param('f_ckpt_model'))
            options['--vae'] = self._make_relative(self._get_param('f_ckpt_vae'))
        elif diffusion_mode == DiffusionMode.UNET:
            options['--diffusion-model'] = self._make_relative(self._get_param('f_unet_model'))
            options['--vae'] = self._make_relative(self._get_param('f_unet_vae'))
            options['--uncond-diffusion-model'] = self._make_relative(self._get_param('f_uncond_unet_model'))
            options['--clip_g'] = self._make_relative(self._get_param('f_clip_g'))
            options['--clip_l'] = self._make_relative(self._get_param('f_clip_l'))
            options['--t5xxl'] = self._make_relative(self._get_param('f_t5xxl'))
            options['--llm'] = self._make_relative(self._get_param('f_llm'))
            options['--llm_vision'] = self._make_relative(self._get_param('f_llm_vision'))

        return {k: v for k, v in options.items() if v is not None}

    def _get_quant_options(self) -> Dict[str, Any]:
        """
        Returns the quantization options shared by CLI and server runners.
        """
        return {
            '--type': (self._get_param('in_model_type')
                       if self._get_param('in_model_type') != "Default"
                       else None),
            '--tensor-type-rules': (
                self._get_param('in_tensor_type_rules')
                if self._get_param('in_tensor_type_rules') != ""
                else None
            ),
        }

    def _get_component_options(self) -> Dict[str, Any]:
        """
        Returns the model component options shared by CLI and server runners.
        """
        threads = self._get_param('in_threads')
        return {
            '-t': (threads
                   if threads and str(threads) != "0"
                   else None),
            '--taesd': self._make_relative(self._get_param('f_taesd')),
            '--photo-maker': (self._make_relative(self._get_param('f_phtmkr'))
                              if self._get_param('in_phtmkr_bool')
                              else None),
            '--pm-id-images-dir': (
                self._make_relative(self._get_param('in_phtmkr_id'))
                if self._get_param('in_phtmkr_bool')
                else None
            ),
            '--pm-id-embed-path': (
                self._make_relative(self._get_param('in_phtmkr_emb'))
                if self._get_param('in_phtmkr_bool')
                else None
            ),
            '--upscale-model': (self._make_relative(self._get_param('f_upscl'))
                                if self._get_param('in_upscl_bool')
                                else None),
            '--control-net': (self._make_relative(self._get_param('f_cnnet'))
                              if self._get_param('in_cnnet_bool')
                              else None),
            '--embd-dir': self._make_relative(config.get('emb_dir')),
            '--prediction': (self._get_param('in_predict')
                             if self._get_param('in_predict') != "Default"
                             else None),
        }

    def _get_performance_options(self) -> Dict[str, Any]:
        """
        Returns the performance/device options shared by CLI and server runners.
        """
        return {
            '--max-vram': self._parse_max_vram_table(),
            '--backend': self._parse_backend_table('in_backend_table'),
            '--params-backend': self._parse_backend_table(
                'in_params_backend_table', params_backend=True
            ),
            '--split-mode': self._parse_split_modes(),
        }

    def _get_common_options(self) -> Dict[str, Any]:
        """
        Returns the runtime options shared by almost all commands.
        """
        rpc_servers = str(self._get_param('in_rpc_servers') or "").strip()
        cache_size = self._get_param('in_conditioning_cache_size')
        linear_scale = self._get_param('in_linear_scale')
        attn_scale = self._get_param('in_attn_scale')
        log_level = self._get_param('in_log_level')
        compression_quality = self._get_param(
            'in_compression_quality'
        )

        return {
            '--auto-fit': ('off' if self._get_param('in_auto_fit') == 'off'
                           else None),
            '--rpc-servers': (rpc_servers if rpc_servers else None),
            '--conditioning-cache-size': (cache_size
                                          if cache_size not in (None, 0)
                                          else None),
            '--linear-scale': (linear_scale
                               if linear_scale not in (None, 0, 0.0)
                               else None),
            '--attn-scale': (attn_scale
                             if attn_scale not in (None, 0, 0.0)
                             else None),
            '--log-level': (log_level
                            if log_level != "info"
                            else None),
            '--compression-quality': (compression_quality
                                      if compression_quality != 90
                                      else None),
        }

    def _get_common_flags(self) -> Dict[str, bool]:
        """
        Returns the execution flags shared by almost all commands.
        """
        return {
            '--eager-load': self._get_param('in_eager_load'),
            '--sage-attn': self._get_param('in_sage_attn'),
            '--disable-prefetch': self._get_param('in_disable_prefetch'),
            '--disable-segmented-compute': (
                self._get_param('in_disable_segmented_compute')
            ),
            '--offload-to-cpu': self._get_param('in_offload_to_cpu'),
            # Deprecated placement compatibility flags
            '--clip-on-cpu': self._get_param('in_clip_on_cpu'),
            '--vae-on-cpu': self._get_param('in_vae_on_cpu'),
            '--control-net-cpu': self._get_param('in_control_net_cpu'),
            '--vae-tiling': self._get_param('in_vae_tiling'),
            '--circular': self._get_param('in_circular_padding') == CIRCULAR_PADDING[1],
            '--circularx': self._get_param('in_circular_padding') == CIRCULAR_PADDING[2],
            '--circulary': self._get_param('in_circular_padding') == CIRCULAR_PADDING[3],
            '--fa': self._get_param('in_flash_attn'),
            '--diffusion-fa': self._get_param('in_diffusion_fa'),
            '--diffusion-conv-direct': self._get_param('in_diffusion_conv_direct'),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '--force-sdxl-vae-conv-scale': self._get_param('in_force_sdxl_vae_conv_scale'),
            '--disable-image-metadata': self._get_param('in_disable_img_metadata'),
            '--mmap': self._get_param('in_mmap'),
            '--color': self._get_param('in_color')
        }

    def _build_process_env(self) -> dict:
        """
        Copies os.environ, injects config env vars,
        prints them, and returns the dict.
        """
        process_env = os.environ.copy()

        if self.env_vars:
            settings_to_print = []
            for key, value in self.env_vars.items():
                if isinstance(value, bool):
                    process_env[key] = "1" if value else "0"
                elif isinstance(value, float) and value.is_integer():
                    process_env[key] = str(int(value))
                else:
                    process_env[key] = str(value)
                settings_to_print.append(f"{key}={process_env[key]}")
            full_line = " ".join(settings_to_print)
            print(f"  SET: {full_line}\n\n")
        return process_env

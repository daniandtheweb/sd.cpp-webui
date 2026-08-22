"""sd.cpp-webui - core - stable-diffusion.cpp common"""

import os
import re
from PIL import Image
from enum import IntEnum
from typing import Dict, Any

from modules.utils.file_utils import get_path
from modules.utils.sdcpp_utils import extract_env_vars, generate_output_filename
from modules.shared_instance import config
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

    def _make_relative(self, path):
        """Returns the path unchanged. Overridden by the CLI runner."""
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
                'in_umt5_xxl', 'in_clip_vision_h', 'in_emb_connect'
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

    def _get_common_flags(self) -> Dict[str, bool]:
        """
        Returns the execution flags shared by almost all commands.
        """
        return {
            '--stream-layers': self._get_param('in_stream_layers'),
            '--eager-load': self._get_param('in_eager_load'),
            '--vae-tiling': self._get_param('in_vae_tiling'),
            '--canny': self._get_param('in_canny'),
            '--chroma-disable-dit-mask': self._get_param('in_disable_dit_mask'),
            '--chroma-enable-t5-mask': self._get_param('in_enable_t5_mask'),
            '--qwen-image-zero-cond-t': self._get_param('in_enable_zero_cond_t'),
            '--circular': self._get_param('in_circular_padding') == CIRCULAR_PADDING[1],
            '--circularx': self._get_param('in_circular_padding') == CIRCULAR_PADDING[2],
            '--circulary': self._get_param('in_circular_padding') == CIRCULAR_PADDING[3],
            '--fa': self._get_param('in_flash_attn'),
            '--diffusion-fa': self._get_param('in_diffusion_fa'),
            '--diffusion-conv-direct': self._get_param('in_diffusion_conv_direct'),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '--force-sdxl-vae-conv-scale': self._get_param('in_force_sdxl_vae_conv_scale'),
            '--mmap': self._get_param('in_mmap'),
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose')
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
                    if value is True:
                        process_env[key] = "1"
                        settings_to_print.append(f"{key}=1")
                elif isinstance(value, int):
                    process_env[key] = str(value)
                    settings_to_print.append(f"{key}={str(value)}")
            if settings_to_print:
                full_line = " ".join(settings_to_print)
                print(f"  SET: {full_line}\n\n")
        return process_env

"""sd.cpp-webui - core - stable-diffusion.cpp common"""

import os
import re
from PIL import Image
from typing import Dict, Any, Optional

from modules.utils.file_utils import get_path
from modules.utils.sdcpp_utils import extract_env_vars, generate_output_filename
from modules.shared_instance import config, SD_CLI
from modules.options import convert, emit, registry

LORA_TAG_PATTERN = re.compile(r'<lora:([^:]+):([^>]+)>')

# Matches %d / %Nd frame index specifiers in output paths (%% is an escaped %).
# Same pattern as sd-cli (stable-diffusion.cpp src/util/image_util.h).
OUTPUT_INDEX_SPECIFIER_REGEX = re.compile(r"(?:[^%]|^)(?:%%)*(%\d{0,3}d)")


def has_output_index_spec(path: str) -> bool:
    """True if the path contains a %d/%Nd frame index specifier (not a %% escape)."""
    return bool(OUTPUT_INDEX_SPECIFIER_REGEX.search(path or ""))


def _printf_index(fmt: str, index: int) -> str:
    """Formats index with a printf-style specifier (%d, %03d, %2d, ...).

    Mirrors snprintf(specifier, frame_idx) for non-negative ints:
    a leading '0' digit is the zero-pad flag (not part of the width),
    the remaining digits are the field width.
    """
    rest = fmt[1:]  # strip the leading '%'
    i = 0
    zero_pad = False
    if rest[0] == '0' and len(rest) > 2 and rest[1].isdigit():
        zero_pad = True
        i = 1
    width = 0
    while i < len(rest) - 1 and rest[i].isdigit():
        width = width * 10 + int(rest[i])
        i += 1
    text = str(index)
    if zero_pad and width > len(text):
        return text.zfill(width)
    return text.rjust(width)


def expand_output_index(path: str, index: int) -> str:
    """
    Expands %d/%Nd frame index specifiers in a path, mirroring sd-cli's
    format_frame_idx(): printf-style padding per specifier (%03d -> zfill,
    %3d -> space pad), then all remaining %% become a literal %.
    """
    result = path or ""
    match = OUTPUT_INDEX_SPECIFIER_REGEX.search(result)
    while match:
        replacement = _printf_index(match.group(1), index)
        # Replace only the specifier (group 1), not the full match, which
        # includes the leading [^%] char — same as C++ (position(1)/length(1)).
        result = result[:match.start(1)] + replacement + result[match.end(1):]
        match = OUTPUT_INDEX_SPECIFIER_REGEX.search(result)
    return result.replace('%%', '%')


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

    def _int_param(self, key: str, default: Any = None) -> Any:
        """
        Helper to get a parameter that must be an integer.

        Gradio Number/Slider widgets always yield floats (e.g. 512.0);
        normalize to int before embedding in CLI args or filenames.
        Returns default (possibly None) when the key is missing.
        """
        val = self.params.get(key, default)
        if val is None:
            return None
        return int(float(val))

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

    @staticmethod
    def _sanitize_output_name(value):
        """Reduces a user-supplied output filename to a safe bare basename.

        Path separators (both OS conventions) and NUL bytes are stripped and
        empty / '.' / '..' results are rejected, so an ``in_output`` value can
        never point outside the output directory. Returns '' when the value
        carries no usable name.
        """
        name = str(value).strip().replace('\\', '/').replace('\x00', '/')
        name = os.path.basename(name)
        if name in ('', '.', '..'):
            return ''
        return name

    def _set_output_path(self, dir_key: str, subctrl_id: int, extension: str):
        """Determines and sets the output path for the command."""
        output_dir = config.get_or_create_dir(dir_key)
        filename_override = self._get_param('in_output')
        output_scheme = config.get('def_output_scheme')

        if filename_override and str(filename_override).strip():
            base_name = self._sanitize_output_name(filename_override)
            if not base_name:
                print(f"Warning: output filename {str(filename_override)!r} is "
                      f"not a valid name; using generated filename instead.")
            else:
                filename = f"{base_name}.{extension}"
                test_path = os.path.join(output_dir, filename)

                counter = 1
                while os.path.exists(test_path):
                    filename = f"{base_name}_{counter}.{extension}"
                    test_path = os.path.join(output_dir, filename)
                    counter += 1

                # Defense in depth: refuse to point outside the output dir
                # even if sanitization were ever weakened again.
                out_root = os.path.realpath(output_dir)
                if not os.path.realpath(test_path).startswith(out_root + os.sep):
                    print(f"Warning: output path {test_path!r} escapes the "
                          f"output directory {out_root!r}; using generated "
                          f"filename instead.")
                else:
                    self.output_path = self._make_relative(test_path)
                    return

        name_parts = []

        if config.get('def_output_steps'):
            steps_val = self._int_param('in_steps')
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
        path_mappings = registry.path_mappings()
        for dir_key, param_keys in path_mappings.items():
            for param_key in param_keys:
                if param_key in self.params:
                    # Create a new key for the full path, e.g., 'f_ckpt_model'
                    full_path_key = f"f_{param_key.replace('in_', '')}"
                    self.params[full_path_key] = get_path(
                        config.get(dir_key), self.params.get(param_key)
                    )

    # Command kind, set by the subclasses (see modules/options/spec.py)
    kind = None

    def _lora_tags(self) -> bool:
        """True if the prompts contain <lora:name:strength> tags."""
        pp_text = str(self._get_param('in_pprompt', "")).strip()
        np_text = str(self._get_param('in_nprompt', "")).strip()
        return bool(LORA_TAG_PATTERN.search(f"{pp_text} {np_text}"))

    def _context(self, values: Optional[Dict[str, Any]] = None,
                 flags: Optional[Dict[str, bool]] = None) -> emit.Context:
        """
        What the option registry needs to build this run's command:
        the values/conditions the runner computes (output path, ...) go
        in 'values' and 'flags'.
        """
        flags = {'lora_tags': self._lora_tags(), **(flags or {})}
        return emit.Context(
            self.params, self.kind, rel=self._make_relative,
            cfg=config.get, values=values, flags=flags
        )

    def _qwen_image_layers(self) -> int | None:
        """The Qwen Image Layered layer count, None when disabled."""
        return convert.qwen_layers(None, self._context())

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

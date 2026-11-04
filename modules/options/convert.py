"""sd.cpp-webui - Option registry - value converters

Named converters used by the registry ('convert="int"'). Each one takes
(value, ctx) and returns the value to output, or None to leave the
option out. 'ctx' gives them the whole parameter dict (ctx.params), so
converters that assemble one value from several settings (tables, mode
specs, key=value lists) are registered the same way.

A converter can take an argument: 'mode_spec:flash_attention'.

This module must not import application modules.
"""

import math
import os
from typing import Any, Callable, Dict, Optional

# Legacy SD-API-style sampler names -> raw sd.cpp sample_method names
SAMPLER_ALIASES = {
    'euler a': 'euler_a',
    'euler': 'euler',
    'ddim': 'ddim_trailing',
    'res multistep': 'res_multistep',
    'res 2s': 'res_2s',
}


def parse_number_list(text: Any, cast=float) -> Optional[list]:
    """
    Parses "1.0, 0.5, 0" or "[7,8,9]" into a list of numbers.
    Returns None if empty or malformed.
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        raw = str(text).strip().strip('[]').replace(';', ',')
        if not raw:
            return None
        items = [p for p in raw.replace(' ', ',').split(',') if p.strip()]
    try:
        return [cast(float(item)) for item in items]
    except (TypeError, ValueError):
        return None


def _num(value: Any, cast) -> Any:
    """int/float from a Gradio value, None if empty or malformed."""
    if value in (None, ''):
        return None
    try:
        # Gradio Number/Slider widgets yield floats (e.g. 512.0)
        return cast(float(value)) if cast is int else cast(value)
    except (ValueError, TypeError):
        return None


def _get(ctx, name: str, default: Any = None) -> Any:
    """Reads the UI setting 'in_<name>'."""
    return ctx.params.get(f'in_{name}', default)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------

def to_int(value, ctx):
    return _num(value, int)


def to_float(value, ctx):
    return _num(value, float)


def to_text(value, ctx):
    """Stripped string, None if empty."""
    if value is None:
        return None
    return str(value).strip() or None


def invert(value, ctx):
    return not value


def seed(value, ctx):
    """The "cleared" seed state (None or '') is sd.cpp's random seed."""
    if value is None or value == '':
        return -1
    return int(float(value))


def int_or_1(value, ctx):
    if value is None:
        return 1
    return int(float(value)) or 1


def strip_quotes(value, ctx):
    """cache-option values may come quoted."""
    text = str(value).strip().strip('"')
    return text or None


def dynamic_policy(value, ctx):
    """scm policy -> scm_policy_dynamic (only dynamic/static count)."""
    text = to_text(value, ctx)
    return text == 'dynamic' if text in ('dynamic', 'static') else None


def tile_square(value, ctx):
    """32 -> '32x32'; 0 / empty -> None."""
    size = _num(value, int)
    return f"{size}x{size}" if size else None


def stem(value, ctx):
    """File stem: the server looks up upscalers by it."""
    if not value:
        return None
    return os.path.splitext(os.path.basename(str(value)))[0] or None


def splitext0(value, ctx):
    """File name without its extension (directories are kept)."""
    return os.path.splitext(str(value))[0]


def sampler_api(value, ctx):
    """UI sampler name -> sd-server sample_method."""
    text = to_text(value, ctx)
    if not text:
        return None
    n = text.lower()
    if n.startswith('k_'):
        n = n[2:]
    return SAMPLER_ALIASES.get(n, n)


def scheduler_api(value, ctx):
    text = to_text(value, ctx)
    if not text:
        return None
    s = text.lower()
    return 'discrete' if s == 'normal' else s


def number_list(value, ctx):
    return parse_number_list(value) or None


def int_list(value, ctx):
    return parse_number_list(value, int) or None


def image_paths(value, ctx):
    """Gradio gallery/file values -> list of paths."""
    if not value:
        return None
    if not isinstance(value, list):
        value = [value]
    paths = []
    for img in value:
        if isinstance(img, tuple):
            paths.append(img[0])
        elif isinstance(img, dict) and "name" in img:
            paths.append(img["name"])
        else:
            paths.append(str(img))
    return paths


# ---------------------------------------------------------------------------
# Server listen address
# ---------------------------------------------------------------------------

def listen_ip(value, ctx):
    return str(value or '127.0.0.1').strip()


def listen_port(value, ctx):
    return _num(value, int) or 1234


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Canonical module names and the aliases the CLI accepts for them
BACKEND_MODULES = {
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
BACKEND_ALIAS_MAP = {
    alias.replace('-', '').replace('_', ''): module
    for module, aliases in BACKEND_MODULES.items()
    for alias in aliases
}
DEFAULT_ALIASES = {'*', 'primary', 'all', 'default'}


def _norm_name(name: str) -> str:
    return str(name).strip().lower().replace('-', '').replace('_', '')


def _backend_table(table: Any, params_backend: bool) -> Optional[str]:
    """
    Parses the backend table rows into a valid sd.cpp backend string.
    Example output: "vulkan0,diffusion=vulkan0&vulkan1,vae=cpu"
    """
    # If UI didn't pass it or it's empty, nothing to output
    if not table or not isinstance(table, list):
        return None

    parts = []
    default_device = None

    for row in table:
        if len(row) < 2:
            continue
        component = _norm_name(row[0])
        device = str(row[1]).strip().lower()

        # Skip invalid/default rows
        if not device or device in ('default', 'auto', ''):
            continue
        # 'disk' is a parameter residency mode, not a compute backend
        if not params_backend and device == 'disk':
            continue

        if component in DEFAULT_ALIASES:
            default_device = device
            continue

        module = BACKEND_ALIAS_MAP.get(component)
        if not module:
            continue
        parts.append(f"{module}={device}")

    # Default entry first, per-module assignments override it
    if default_device:
        entry = f"*={default_device}" if params_backend else default_device
        parts.insert(0, entry)

    return ",".join(parts) if parts else None


def backend_table(value, ctx):
    return _backend_table(value, params_backend=False)


def params_backend_table(value, ctx):
    return _backend_table(value, params_backend=True)


# Only these modules support layer/row splitting of their compute
SPLIT_MODULES = {'diffusion', 'te'}


def split_modes(value, ctx):
    """
    Per-module split mode rows -> '--split-mode' value.
    Example output: "diffusion=row,te=layer"
    """
    if not value or not isinstance(value, list):
        return None
    parts = []
    for row in value:
        if not row or len(row) < 2:
            continue
        module = str(row[0]).strip().lower()
        mode = str(row[1]).strip().lower()
        if module in SPLIT_MODULES and mode in ('layer', 'row'):
            parts.append(f"{module}={mode}")
    return ",".join(parts) if parts else None


# Budget keys the CLI accepts in place of a device name
MAX_VRAM_DEFAULT_KEYS = {'', 'default', 'all', '*'}


def max_vram(value, ctx):
    """
    Max-vram table rows -> sd.cpp budget string.
    Example output: "cuda0=6,vulkan0=2". Budgets of 0 are left out, and
    so is the whole option when none is left.
    """
    if not value or not isinstance(value, list):
        return None
    parts = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        device = str(row[0] or '').strip()
        budget_text = str(row[1] or '').strip()
        if not budget_text or device.lower() == 'disk':
            continue
        try:
            budget = float(budget_text)
        except ValueError:
            continue
        if not math.isfinite(budget) or budget == 0:
            continue
        if device.lower() in MAX_VRAM_DEFAULT_KEYS:
            parts.append(budget_text)
        else:
            parts.append(f"{device}={budget_text}")
    return ",".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Aggregates: one value assembled from several settings
# ---------------------------------------------------------------------------

def mode_spec(value, ctx, base: str = ''):
    """
    on|off|auto value with per-module overrides, as used by --conv-direct
    and --flash-attn, e.g. "auto,vae=off". Reads 'in_<base>' (global
    mode), '<base>_diffusion' and '<base>_vae'. A module set to "auto"
    follows the global mode. None when everything is "auto".
    """
    modes = ("auto", "on", "off")

    def mode(name):
        val = str(_get(ctx, name) or "auto").lower()
        return val if val in modes else "auto"

    parts = [mode(base)]
    for module in ("diffusion", "vae"):
        module_mode = mode(f"{base}_{module}")
        if module_mode != "auto":
            parts.append(f"{module}={module_mode}")
    spec = ",".join(parts)
    return None if spec == "auto" else spec


def model_args(value, ctx):
    """The --model-args key=value string, from the Model Arguments tab."""
    args = []
    if not _get(ctx, 'chroma_use_dit_mask', True):
        args.append('chroma_use_dit_mask=false')
    if _get(ctx, 'chroma_use_t5_mask'):
        args.append('chroma_use_t5_mask=true')
    t5_mask_pad = _get(ctx, 'chroma_t5_mask_pad')
    if t5_mask_pad and float(t5_mask_pad) != 0:
        args.append(f'chroma_t5_mask_pad={int(t5_mask_pad)}')
    if _get(ctx, 'qwen_image_zero_cond_t'):
        args.append('qwen_image_zero_cond_t=true')
    if _get(ctx, 'qwen_image_2_1_prefix_cache'):
        args.append('qwen_image_2_1_prefix_cache=true')
    prefix_cache_type = _get(ctx, 'qwen_image_2_1_prefix_cache_type')
    if prefix_cache_type and prefix_cache_type != "auto":
        args.append(f'qwen_image_2_1_prefix_cache_type={prefix_cache_type}')
    base_size = _get(ctx, 'pixart_pos_embed_base_size')
    if base_size and float(base_size) != 0:
        args.append(f'pixart_pos_embed_base_size={int(base_size)}')
    scale = _get(ctx, 'pixart_interpolation_scale')
    if scale and float(scale) != 0:
        args.append(f'pixart_interpolation_scale={scale}')
    factor = _get(ctx, 'pixart_vae_scale_factor')
    if factor and float(factor) != 0:
        args.append(f'pixart_vae_scale_factor={factor}')
    return ",".join(args) or None


# ADetailer settings: (UI name, extra_ad_args key, type, sd.cpp default)
AD_ARGS = (
    ('ad_confidence', 'confidence', float, 0.3),
    ('ad_denoising_strength', 'denoising_strength', float, 0.4),
    ('ad_mask_blur', 'mask_blur', int, 4),
    ('ad_inpaint_padding', 'inpaint_padding', int, 32),
    ('ad_dilate_erode', 'dilate_erode', int, 4),
    ('ad_mask_mode', 'mask_mode', str, "none"),
    ('ad_inpaint_width', 'inpaint_width', int, 512),
    ('ad_inpaint_height', 'inpaint_height', int, 512),
    ('ad_steps', 'steps', int, 0),
    ('ad_cfg', 'cfg_scale', float, -1.0),
)


def ad_args(value, ctx):
    """
    The extra ADetailer key=value list, from the ADetailer settings,
    skipping the values equal to sd.cpp's defaults. The free-form extra
    arguments come last, so they take precedence.
    """
    args = []
    for name, key, cast, default in AD_ARGS:
        val = _get(ctx, name)
        if val is None or str(val).strip() == "":
            continue
        try:
            val = int(float(val)) if cast is int else cast(val)
        except (TypeError, ValueError):
            continue
        if val == default:
            continue
        if cast is float:
            val = f"{val:g}"
        args.append(f"{key}={val}")
    extra = str(_get(ctx, 'ad_extra_args') or "").strip(" ,")
    if extra:
        args.append(extra)
    return ",".join(args) or None


def ref_image_args(value, ctx):
    """The --ref-image-args list; None when nothing is set (auto-detected)."""
    args = []
    if _get(ctx, 'increase_ref_index'):
        args.append("ref_index_mode=increase")
    if _get(ctx, 'disable_auto_resize_ref_image'):
        args.append("resize_before_vae=false")
    return ",".join(args) or None


def qwen_layers(value, ctx):
    """Qwen Image Layered layer count, None when disabled."""
    if not _get(ctx, 'qwen_layered_bool'):
        return None
    layers = _get(ctx, 'qwen_image_layers')
    if layers is None:
        return None
    return max(0, int(float(layers)))


def api_preview_mode(value, ctx):
    """sd-server preview mode. Tabs without the settings use 'proj'."""
    if 'in_preview_bool' not in ctx.params:
        return "proj"
    if not _get(ctx, 'preview_bool'):
        return "none"
    return to_text(_get(ctx, 'preview_mode'), ctx) or "proj"


def api_preview_interval(value, ctx):
    if 'in_preview_bool' not in ctx.params:
        return 1
    if not _get(ctx, 'preview_bool'):
        return 1
    return max(1, _num(_get(ctx, 'preview_interval'), int) or 1)


CONVERTERS: Dict[str, Callable] = {
    'int': to_int,
    'float': to_float,
    'text': to_text,
    'not': invert,
    'seed': seed,
    'int_or_1': int_or_1,
    'strip_quotes': strip_quotes,
    'dynamic_policy': dynamic_policy,
    'tile_square': tile_square,
    'stem': stem,
    'splitext0': splitext0,
    'sampler_api': sampler_api,
    'scheduler_api': scheduler_api,
    'number_list': number_list,
    'int_list': int_list,
    'image_paths': image_paths,
    'listen_ip': listen_ip,
    'listen_port': listen_port,
    'backend_table': backend_table,
    'params_backend_table': params_backend_table,
    'split_modes': split_modes,
    'max_vram': max_vram,
    'mode_spec': mode_spec,
    'model_args': model_args,
    'ad_args': ad_args,
    'ref_image_args': ref_image_args,
    'qwen_layers': qwen_layers,
    'api_preview_mode': api_preview_mode,
    'api_preview_interval': api_preview_interval,
}


def convert(name: str, value: Any, ctx) -> Any:
    """Applies the named converter ('name' or 'name:argument')."""
    name, _, arg = name.partition(':')
    fn = CONVERTERS[name]
    return fn(value, ctx, arg) if arg else fn(value, ctx)

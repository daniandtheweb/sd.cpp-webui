"""The settings of the upscale: what they are, where they are kept, and
the parameters of the generation they make for every tile.
"""

import json
import os

from . import tiles

FILE = "settings.json"

SAME = "Same as the image"
CUSTOM = "Custom"
NO_UPSCALER = "None (resize only)"
CHECKPOINT, UNET = "Checkpoint", "UNET"

# What a generation takes from the model of the image, when the upscale
# uses the same one (the keys of ctx.params)
MODEL_KEYS = (
    'in_diffusion_mode', 'in_ckpt_model', 'in_ckpt_vae', 'in_unet_model',
    'in_unet_vae', 'in_clip_g', 'in_clip_l', 'in_t5xxl', 'in_llm',
    'in_llm_vision', 'in_tokenizer', 'in_umt5_xxl', 'in_clip_vision_h',
    'in_model_type',
)


def defaults(api) -> dict:
    """The settings of a new install; steps and CFG start as the app's."""
    base = api.default_params()
    return {
        # Upscale
        'scale': 2.0,
        'upscaler': NO_UPSCALER,
        # Model
        'model_from': SAME,
        'model_type': CHECKPOINT,
        'checkpoint': base.get('in_ckpt_model') or "",
        'vae': base.get('in_ckpt_vae') or "",
        'unet': "", 'unet_vae': "", 'clip_l': "", 'clip_g': "",
        't5xxl': "", 'llm': "",
        # Prompts: empty is the prompt of the image
        'prompt': "", 'negative': "",
        # Sampling
        'sampler': SAME, 'scheduler': SAME,
        'steps': int(base.get('in_steps') or 20),
        'cfg': float(base.get('in_cfg') or 7.0),
        'seed': -1,
        'denoise': 0.3,
        # Tiles
        'mode': tiles.LINEAR,
        'tile_width': 512, 'tile_height': 512,
        'blur': 8, 'padding': 32,
        'uniform': True, 'tiled_decode': False, 'live_preview': True,
        # Seam fix
        'seam': tiles.NO_SEAMS, 'seam_denoise': 0.35, 'seam_width': 64,
        'seam_blur': 8, 'seam_padding': 32,
        # ControlNet, for the tile model
        'controlnet': False, 'cn_model': "", 'cn_strength': 1.0,
        'cn_union': False, 'cn_type': "tile",
    }


def _path(api) -> str:
    return os.path.join(api.data_dir(), FILE)


def load(api) -> dict:
    """The saved settings over the defaults; a wrong one is ignored."""
    values = defaults(api)
    try:
        with open(_path(api), encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return values
    if isinstance(saved, dict):
        for key, value in saved.items():
            if key in values and type(value) is type(values[key]):
                values[key] = value
            elif (key in values and isinstance(value, (int, float))
                  and isinstance(values[key], (int, float))
                  and not isinstance(value, bool)
                  and not isinstance(values[key], bool)):
                values[key] = type(values[key])(value)
    return values


def save(api, values: dict):
    """Keep the settings between runs."""
    known = defaults(api)
    clean = {k: v for k, v in values.items() if k in known}
    with open(_path(api), 'w', encoding='utf-8') as f:
        json.dump(clean, f, indent=2)


def stages(cfg: dict, size: tuple[int, int]) -> list:
    """The stages of the redraw for an image of this size."""
    return tiles.stages(
        size, int(cfg['tile_width']), int(cfg['tile_height']),
        mode=cfg['mode'], padding=int(cfg['padding']),
        blur=int(cfg['blur']), denoise=float(cfg['denoise']),
        uniform=bool(cfg['uniform']), seam=cfg['seam'],
        seam_width=int(cfg['seam_width']),
        seam_padding=int(cfg['seam_padding']),
        seam_blur=int(cfg['seam_blur']),
        seam_denoise=float(cfg['seam_denoise']))


def _model(cfg: dict, image_params: dict) -> dict:
    """The model keys of a generation: the image's or the chosen ones."""
    if cfg['model_from'] == SAME:
        return {k: image_params[k] for k in MODEL_KEYS if k in image_params}
    if cfg['model_type'] == UNET:
        chosen = {
            'in_diffusion_mode': 1, 'in_unet_model': cfg['unet'],
            'in_unet_vae': cfg['unet_vae'], 'in_clip_l': cfg['clip_l'],
            'in_clip_g': cfg['clip_g'], 'in_t5xxl': cfg['t5xxl'],
            'in_llm': cfg['llm'],
        }
    else:
        chosen = {
            'in_diffusion_mode': 0, 'in_ckpt_model': cfg['checkpoint'],
            'in_ckpt_vae': cfg['vae'],
        }
    return {k: v or None for k, v in chosen.items()} | {
        'in_diffusion_mode': chosen['in_diffusion_mode']}


PREVIEW_KEYS = ('in_preview_mode', 'in_preview_interval',
                'in_preview_taesd', 'in_preview_noisy')


def _preview(cfg: dict, image_params: dict) -> dict:
    """
    What a tile is generated with to be seen while it works: the preview
    settings of the tab if it has them on, else the cheap projection.
    """
    if not cfg['live_preview']:
        return {'in_preview_bool': False}
    if image_params.get('in_preview_bool'):
        return {'in_preview_bool': True} | {
            k: image_params[k] for k in PREVIEW_KEYS if k in image_params}
    return {'in_preview_bool': True, 'in_preview_mode': 'proj',
            'in_preview_interval': 1}


def tile_params(api, cfg: dict, image_params: dict, seed: int) -> dict:
    """
    What every tile is generated with: the defaults of the app, the model
    of the image or the chosen one, and the settings of the upscale. The
    image, its size and the denoise are set for each tile.
    """
    params = api.default_params()
    params.update(_model(cfg, image_params))
    params.update({
        'in_pprompt': cfg['prompt'] or image_params.get('in_pprompt') or "",
        'in_nprompt': cfg['negative'] or image_params.get('in_nprompt') or "",
        'in_steps': int(cfg['steps']), 'in_cfg': float(cfg['cfg']),
        'in_seed': seed, 'in_batch_count': 1,
        'in_img_mask': None,
        'in_upscl_bool': False, 'in_hires_bool': False, 'in_output': "",
        **_preview(cfg, image_params),
        'in_vae_tiling': bool(cfg['tiled_decode']),
        'in_cnnet_bool': bool(cfg['controlnet'] and cfg['cn_model']),
    })
    for key, name in (('in_sampling', 'sampler'),
                      ('in_scheduler', 'scheduler')):
        if cfg[name] != SAME:
            params[key] = cfg[name]
        elif image_params.get(key):
            params[key] = image_params[key]
    if params['in_cnnet_bool']:
        params.update({
            'in_cnnet': cfg['cn_model'],
            'in_control_strength': float(cfg['cn_strength']),
            'in_cnnet_union': bool(cfg['cn_union']),
            'in_control_type': cfg['cn_type'],
        })
    return params


SDCPP_JSON = ", SDCPP: "      # the app's own record, always the last


def usage_text(cfg: dict, params: dict, size: tuple, seed: int,
               original: str = "", note: str = "") -> str:
    """The upscale as fields of the generation info (Key: value, ...)."""
    fields = [
        f"Ultimate SDCPP Upscale: x{cfg['scale']}",
        f"Upscale size: {size[0]}x{size[1]}",
        f"Upscale upscaler: {cfg['upscaler']}",
        f"Upscale seed: {seed}",
        f"Upscale steps: {cfg['steps']}",
        f"Upscale CFG scale: {cfg['cfg']}",
        f"Upscale denoising strength: {cfg['denoise']}",
        f"Upscale mode: {cfg['mode']}",
        f"Upscale tile: {cfg['tile_width']}x{cfg['tile_height']}",
        f"Upscale padding: {cfg['padding']}",
        f"Upscale blur: {cfg['blur']}",
        f"Upscale seam fix: {cfg['seam']}",
    ]
    if cfg['seam'] != tiles.NO_SEAMS:
        fields += [
            f"Upscale seam denoising strength: {cfg['seam_denoise']}",
            f"Upscale seam width: {cfg['seam_width']}",
            f"Upscale seam padding: {cfg['seam_padding']}",
            f"Upscale seam blur: {cfg['seam_blur']}",
        ]
    if params.get('in_cnnet_bool'):
        fields += [
            f"Upscale ControlNet: {params['in_cnnet']}",
            f"Upscale ControlNet strength: {params['in_control_strength']}",
        ]
    for key, name in (('in_sampling', "sampler"),
                      ('in_scheduler', "scheduler")):
        if params.get(key):         # the one used, also if it is the image's
            fields.append(f"Upscale {name}: {params[key]}")
    for name in ("prompt", "negative"):
        if cfg[name]:
            fields.append(f"Upscale {name}: {json.dumps(cfg[name])}")
    if original:
        fields.append(f"Upscale original: {original}")
    if note:
        fields.append(f"Upscale note: {note}")
    return ", ".join(fields)


def server_needs(params: dict) -> dict:
    """
    What the tiles need the server to have loaded when it starts (the
    models and the ControlNet), from the params of a tile.
    """
    needs = {key: params[key] for key in MODEL_KEYS if params.get(key)}
    if params.get('in_cnnet_bool'):
        needs['in_cnnet'] = params['in_cnnet']
    return needs


def parameters_text(cfg: dict, params: dict, size: tuple, seed: int,
                    original: str = "", note: str = "",
                    source: str = "") -> str:
    """
    The generation info saved in the image: the one of the image that was
    upscaled, as it was, with the upscale added as more fields (before the
    app's own record, which has to stay the last). Without the info of the
    image it is the upscale's own, like the app's.
    """
    usage = usage_text(cfg, params, size, seed, original, note)
    if source.strip():
        head, found, record = source.rstrip().rpartition(SDCPP_JSON)
        if found and record.lstrip().startswith("{"):
            return f"{head}, {usage}{found}{record}"
        return f"{source.rstrip()}, {usage}"
    lines = [str(params.get('in_pprompt') or "")]
    if params.get('in_nprompt'):
        lines.append(f"Negative prompt: {params['in_nprompt']}")
    lines.append(
        f"Steps: {cfg['steps']}, CFG scale: {cfg['cfg']}, Seed: {seed}, "
        f"Size: {size[0]}x{size[1]}, Denoising strength: {cfg['denoise']}, "
        + usage)
    return "\n".join(lines)

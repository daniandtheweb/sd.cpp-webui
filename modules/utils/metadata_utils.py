"""sd.cpp-webui - utils - metadata generation and parsing module"""

import os
import re
import json
from typing import Any, Dict, Optional
from PIL import Image, PngImagePlugin

from modules.shared_instance import server_state


def build_a1111_metadata(
    params: Dict[str, Any], seed: int, default_vae: str = 'unknown'
) -> str:
    """
    Constructs an A1111-formatted metadata string.
    """
    pprompt = params.get_param('in_pprompt', '')
    nprompt = params.get_param('in_nprompt', '')
    steps = params.get_param('in_steps', 20)
    cfg = params.get_param('in_cfg', 7.0)
    seed = server_state.seed
    width = params.get_param('in_width', 512)
    height = params.get_param('in_height', 512)
    sampler = params.get_param('in_sampling', 'Euler a')
    scheduler = params.get_param('in_scheduler', '')
    rng = params.get_param('in_rng', '')
    sampler_rng = params.get_param('in_sampler_rng', '')

    model_path = str(params.get_param('in_ckpt_model', 'unknown'))
    model_name = os.path.basename(model_path)

    vae_path = str(params.get_param('in_ckpt_vae', 'unknown'))
    vae_name = os.path.basename(vae_path)

    full_sampler = f"{sampler} {scheduler}".strip()

    meta_str = f"{pprompt}\n"
    if nprompt:
        meta_str += f"Negative prompt: {nprompt}\n"

    meta_str += (
        f"Steps: {steps}, CFG scale: {cfg}, Seed: {seed}, "
        f"Size: {width}x{height}, Model: {model_name}, "
        f"RNG: {rng}, Sampler RNG: {sampler_rng} , "
        f"Sampler: {full_sampler}, "
        f"VAE: {vae_name}, Version: sd.cpp-webui"
    )

    return meta_str


def save_image_with_metadata(
    image: Image.Image, target_path: str, metadata_string: str
):
    """
    Saves a PIL Image with A1111-style parameters
    written to the tEXt chunk.
    """
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", metadata_string)
    image.save(target_path, format="PNG", pnginfo=pnginfo)


def parse_png_metadata(img_path: str) -> Optional[str]:
    """Reads the tEXt chunk from a PNG file to get metadata."""
    try:
        with open(img_path, 'rb') as file:
            if file.read(8) != b'\x89PNG\r\n\x1a\n':
                return None  # Not a valid PNG

            while True:
                length_chunk = file.read(4)
                if not length_chunk:
                    break
                length = int.from_bytes(length_chunk, byteorder='big')
                chunk_type = file.read(4).decode('utf-8', errors='ignore')

                if chunk_type == 'tEXt':
                    # Found the metadata chunk
                    png_block = file.read(length)
                    _ = file.read(4)  # Skip CRC
                    keyword, value = png_block.split(b'\x00', 1)
                    return (
                        f"PNG: tEXt\n{keyword.decode('utf-8')}: "
                        f"{value.decode('utf-8')}"
                    )

                file.seek(length + 4, 1)  # Skip chunk data and CRC

    except Exception:
        return None
    return None


def parse_jpg_metadata(img_path: str) -> Optional[str]:
    """Extracts UserComment EXIF data from a JPG file."""
    try:
        with Image.open(img_path) as img:
            exif_data = img._getexif()
            if exif_data:
                exif = exif_data.get(37510)  # 37510 = UserComment tag
                if exif:
                    return (
                        f"JPG: Exif\nPositive prompt: "
                        f"{exif.decode('utf-8', errors='ignore')[8:]}"
                    )
                return "JPG: No User Comment found."
    except Exception:
        return "JPG: No EXIF data found."
    return "JPG: No EXIF data found."


def parse_comfyui_workflow(text_data: str) -> Optional[Dict[str, Any]]:
    """
    Parses a ComfyUI JSON workflow using match-case
    and a data-driven approach.
    """
    try:
        json_start_index = text_data.find('{')
        if json_start_index == -1:
            return None

        workflow_data = json.loads(text_data[json_start_index:])
        params = {}
        found_positive = False

        keys_to_extract = ["steps", "cfg", "sampler_name", "scheduler"]

        for node in workflow_data.values():
            if not isinstance(node, dict):
                continue

            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            match class_type:
                case "CLIPTextEncode":
                    meta_title = (
                        node.get("_meta", {}).get("title", "").lower()
                    )
                    if "positive" in meta_title and not found_positive:
                        params['pprompt'] = inputs.get("text")
                        found_positive = True
                    elif "negative" in meta_title:
                        params['nprompt'] = inputs.get("text")

            for key in keys_to_extract:
                if key in inputs:
                    params[key] = inputs[key]

            if "noise_seed" in inputs:
                params['seed'] = inputs["noise_seed"]
            elif "seed" in inputs:
                params['seed'] = inputs["seed"]

            if "sampler_name" in inputs:
                params['sampler'] = inputs['sampler_name']

        return params if any(params.values()) else None

    except (json.JSONDecodeError, AttributeError):
        return None


def parse_a1111_text(text_data: str) -> Dict[str, Any]:
    """
    Parses A1111-style text metadata, now separating sampler and scheduler
    based on the first space.
    """
    params = {}

    patterns = {
        'pprompt': (
            r'(?s)(?:Positive prompt|parameters):\s*(.*?)'
            r'(?=\s*(?:Negative prompt:|Steps:|CFG scale:|Seed:|'
            r'Size:|Model:|Sampler:|$))'
        ),
        'nprompt': (
            r'(?s)Negative prompt:\s*(.*?)'
            r'(?=\s*(?:Steps:|CFG scale:|Seed:|Size:|Model:|Sampler:|$))'
        ),
        'steps': r'Steps:\s*(\d+)',
        'cfg': r'CFG scale:\s*([\d.]+)',
        'seed': r'Seed:\s*(\d+)',
    }
    converters = {'steps': int, 'cfg': float, 'seed': int}

    for key, pattern in patterns.items():
        match = re.search(
            pattern,
            text_data,
            re.IGNORECASE
        )
        if match:
            value = match.group(1).strip()
            if key not in ['pprompt', 'nprompt']:
                value = value.split(',')[0]

            params[key] = converters.get(key, lambda x: x)(value)

    sampler_match = re.search(
        r'Sampler:\s*([^,]+)', text_data, re.IGNORECASE
    )
    if sampler_match:
        full_sampler_str = sampler_match.group(1).strip()
        parts = full_sampler_str.split(' ', 1)
        params['sampler'] = parts[0]
        if len(parts) > 1:
            params['scheduler'] = parts[1]
        else:
            params['scheduler'] = ""

    return params


def extract_params_from_text(text_data: str) -> Dict[str, Any]:
    """
    Extracts generation parameters by dispatching to the correct parser.
    """
    default_params = {
        'pprompt': "", 'nprompt': "", 'steps': None, 'sampler': "",
        'scheduler': "", 'cfg': None, 'seed': None
    }

    if not text_data:
        return default_params

    comfy_params = parse_comfyui_workflow(text_data)
    if comfy_params:
        return {**default_params, **comfy_params}

    a1111_params = parse_a1111_text(text_data)
    return {**default_params, **a1111_params}

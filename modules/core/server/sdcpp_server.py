"""sd.cpp-webui - core - stable-diffusion.cpp server"""

import os
import io
import re
import json
import base64
import requests
from PIL import Image
from typing import Dict, Any, Generator

import gradio as gr

from modules.utils.file_utils import get_path
from modules.utils.sdcpp_utils import generate_output_filename
from modules.utils.metadata_utils import (
    build_a1111_metadata, save_image_with_metadata
)
from modules.shared_instance import config, server_state


class ApiTaskRunner:
    """
    Builds and manages API requests to the sd.cpp server.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.ip = str(self._get_param('in_ip'))
        self.port = str(self._get_param('in_port'))
        self.url = ""

        self.output_path = ""
        self.outputs = []
        self.fcommand = ""

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
                    full_path_key = f"f_{param_key.replace('in_', '')}"
                    self.params[full_path_key] = get_path(
                        config.get(dir_key), self.params.get(param_key)
                    )

    def _extract_loras(self, text: str) -> tuple:
        """
        Finds <lora:name:multiplier> tags, extracts them into dicts,
        and removes them from the text.
        Returns: (cleaned_text, list_of_lora_dicts)
        """
        if not text:
            return text, []

        extracted_loras = []
        # Regex matches <lora:filename:multiplier>
        pattern = r'<lora:([^:]+):([^>]+)>'

        def match_handler(match):
            path = match.group(1).strip()
            try:
                multiplier = float(match.group(2).strip())
            except ValueError:
                multiplier = 1.0

            extracted_loras.append({
                "path": path,
                "multiplier": multiplier,
                "is_high_noise": False
            })
            return ""

        cleaned_text = re.sub(pattern, match_handler, text)

        return cleaned_text, extracted_loras

    def _get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _get_extra_args_string(self) -> str:
        """Helper to build the <sd_cpp_extra_args> tag for the prompt."""
        mapping = {
            'in_steps': ('steps', int, None),
            'in_seed': ('seed', int, None),
            'in_cfg': ('cfg_scale', float, None),
            'in_strength': ('denoising_strength', float, None),
            'in_flow_shift': ('flow_shift', float, 'in_flow_shift_bool'),
            'in_guidance': ('guidance', float, 'in_guidance_bool'),
            'in_clip_skip': ('clip_skip', int, None),
            'in_eta': ('eta', float, 'in_eta_bool'),
            'in_timestep_shift': ('timestep_shift', float, 'in_timestep_shift_bool'),
            'in_sampling': ('sampler_name', str, None),
            'in_scheduler': ('scheduler', str, None),
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

        return f"<sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>" if extra_args else ""

    def _build_payload(self) -> dict:
        """Constructs the JSON payload for sdapi endpoints."""
        raw_pprompt = self._get_param('in_pprompt', '')
        raw_nprompt = self._get_param('in_nprompt', '')

        clean_pprompt, pprompt_loras = self._extract_loras(raw_pprompt)
        clean_nprompt, nprompt_loras = self._extract_loras(raw_nprompt)

        loras = pprompt_loras + nprompt_loras

        payload = {
            "prompt": clean_pprompt,
            "negative_prompt": clean_nprompt,
            "sampler_name": self._get_param('in_sampling', 'Euler a'),
            "scheduler": self._get_param('in_scheduler', 'discrete'),
            "width": int(self._get_param('in_width', 512)),
            "height": int(self._get_param('in_height', 512)),
            "batch_size": int(self._get_param('in_batch_count', 1)),
            "seed": int(self._get_param('in_seed', -1)),
            "steps": int(self._get_param('in_steps', 20)),
            "cfg_scale": float(self._get_param('in_cfg', 7.0)),
        }

        if loras:
            unique_loras = {lora['path']: lora for lora in loras}.values()
            payload["lora"] = list(unique_loras)

        return payload

    def _process_response(self, data: dict):
        base, ext = os.path.splitext(self.output_path)
        images_list = data.get("images", []) or [item.get("b64_json") for item in data.get("data", []) if "b64_json" in item]

        for i, b64_data in enumerate(images_list):
            if not b64_data:
                continue
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]

            image_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(image_bytes))
            target_path = self.output_path if i == 0 else f"{base}_{i + 1}{ext}"
            if ext.lower() == '.png':
                meta_string = build_a1111_metadata(self.params, server_state.seed)
                save_image_with_metadata(image, target_path, meta_string)
            else:
                image.save(target_path)
            self.outputs.append(target_path)

    def run(self) -> Generator:
        self._resolve_paths()
        payload_or_files = self._build_payload()

        if isinstance(payload_or_files, tuple):
            self.fcommand = json.dumps(payload_or_files[0], indent=4)
        else:
            display_payload = payload_or_files.copy()
            for key in ["init_images", "mask"]:
                if key in display_payload:
                    if isinstance(display_payload[key], list):
                        display_payload[key] = [img[:60] + "..." for img in display_payload[key]]
                    elif isinstance(display_payload[key], str):
                        display_payload[key] = display_payload[key][:60] + "..."
            self.fcommand = json.dumps(display_payload, indent=4)

        yield (
            self.fcommand,
            gr.skip(),
            gr.skip(),
            gr.skip(),
            None
        )

        try:
            if isinstance(payload_or_files, tuple):
                data, files = payload_or_files
                response = requests.post(
                    self.url, data=data, files=files, timeout=None
                )
            else:
                response = requests.post(
                    self.url, json=payload_or_files, timeout=None
                )

            if response.status_code == 200:
                self._process_response(response.json())
                gen_stats = getattr(
                    server_state, "last_generation_stats", "No stats recorded."
                )

                yield (
                    self.fcommand,
                    gr.skip(),
                    gr.skip(),
                    gr.update(value=gen_stats),
                    self.outputs
                )
            else:
                yield (
                    self.fcommand,
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    None
                )
        except Exception:
            import traceback
            traceback.print_exc()
            yield (
                self.fcommand,
                gr.skip(),
                gr.skip(),
                gr.skip(),
                None
            )


class Txt2ImgApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(
            dir_key='txt2img_dir', subctrl_id=0, extension='png'
        )
        self.url = f"http://{self.ip}:{self.port}/sdapi/v1/txt2img"


class Img2ImgApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(
            dir_key='img2img_dir', subctrl_id=0, extension='png'
        )
        self.url = f"http://{self.ip}:{self.port}/sdapi/v1/img2img"

    def _build_payload(self) -> dict:
        payload = super()._build_payload()
        init_img = self._get_param('in_img_inp') or self._get_param('in_first_frame_inp')
        if init_img is not None:
            if isinstance(init_img, str):
                init_img = Image.open(init_img)
            elif not isinstance(init_img, Image.Image):
                init_img = Image.fromarray(init_img)

            buf = io.BytesIO()
            init_img.save(buf, format="PNG")
            payload["init_images"] = [base64.b64encode(buf.getvalue()).decode('utf-8')]

        mask_img = self._get_param('in_mask_img')
        if mask_img is not None:
            if not isinstance(mask_img, Image.Image):
                mask_img = Image.fromarray(mask_img)
            m_buf = io.BytesIO()
            mask_img.save(m_buf, format="PNG")
            payload["mask"] = base64.b64encode(m_buf.getvalue()).decode('utf-8')
            payload["inpainting_mask_invert"] = self._get_param('in_invert_mask', False)

        return payload


class ImgEditApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(dir_key='imgedit_dir', subctrl_id=2, extension='png')
        self.url = f"http://{self.ip}:{self.port}/v1/images/edits"

    def _build_payload(self) -> tuple:
        """Constructs multipart form data and files."""
        extra_args_str = self._get_extra_args_string()
        prompt = self._get_param('in_pprompt', '') + extra_args_str

        form_data = {
            "prompt": prompt,
            "n": str(self._get_param('in_batch_count', 1)),
            "size": f"{self._get_param('in_width')}x{self._get_param('in_height')}",
            "sampler_name": self._get_param('in_sampling', 'Euler a'),
            "scheduler": self._get_param('in_scheduler', 'discrete'),
            "output_format": "png"
        }

        # Files
        files = []
        init_img = self._get_param('in_ref_img') or self._get_param('in_img_inp')
        if init_img is not None:
            if isinstance(init_img, str):
                init_img = Image.open(init_img)
            elif not isinstance(init_img, Image.Image):
                init_img = Image.fromarray(init_img)

            buf = io.BytesIO()
            init_img.save(buf, format="PNG")
            buf.seek(0)
            files.append(("image[]", ("image.png", buf, "image/png")))

        mask_img = self._get_param('in_mask_img')
        if mask_img is not None:
            if isinstance(mask_img, str):
                mask_img = Image.open(mask_img)
            elif not isinstance(mask_img, Image.Image):
                mask_img = Image.fromarray(mask_img)

            m_buf = io.BytesIO()
            mask_img.save(m_buf, format="PNG")
            m_buf.seek(0)
            files.append(("mask", ("mask.png", m_buf, "image/png")))

        return form_data, files


def txt2img_api(params: dict):
    runner = Txt2ImgApiRunner(params)
    runner.prepare()
    yield from runner.run()


def img2img_api(params: dict):
    runner = Img2ImgApiRunner(params)
    runner.prepare()
    yield from runner.run()


def imgedit_api(params: dict):
    runner = ImgEditApiRunner(params)
    runner.prepare()
    yield from runner.run()

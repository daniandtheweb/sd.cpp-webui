"""sd.cpp-webui - core - stable-diffusion.cpp server"""

import os
import io
import json
import time
import base64
import re
import threading
import httpx
from PIL import Image, ImageOps
from typing import Dict, Any, Generator

import gradio as gr

from modules.core.common.sd_common import (
    CommonRunner, LORA_TAG_PATTERN, image_to_pil, process_editor_mask,
    has_output_index_spec, expand_output_index
)
from modules.loader import get_loras
from modules.utils.metadata_utils import (
    build_a1111_metadata, save_image_with_metadata
)
from modules.shared_instance import config, server_state


class ApiTaskRunner(CommonRunner):
    """
    Builds and manages API requests to the sd.cpp server.
    """

    _lora_lookup_cache = {}

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.ip = str(self._get_param('in_ip'))
        self.port = str(self._get_param('in_port'))
        self.url = ""
        self.use_async = False

        self.output_path = ""
        self.outputs = []

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
        matches = LORA_TAG_PATTERN.findall(text)
        needs_refresh = False

        for name, _ in matches:
            name = name.strip()

            if name not in ApiTaskRunner._lora_lookup_cache:
                needs_refresh = True
                break

        if needs_refresh:
            available_loras = get_loras()

            ApiTaskRunner._lora_lookup_cache.clear()

            for lora_path in available_loras:
                base_name = os.path.splitext(os.path.basename(lora_path))[0]
                path_no_ext = os.path.splitext(lora_path)[0]
                base_with_ext = os.path.basename(lora_path)

                ApiTaskRunner._lora_lookup_cache[base_name] = lora_path
                ApiTaskRunner._lora_lookup_cache[path_no_ext] = lora_path
                ApiTaskRunner._lora_lookup_cache[base_with_ext] = lora_path
                ApiTaskRunner._lora_lookup_cache[lora_path] = lora_path

        def match_handler(match):
            parsed_name = match.group(1).strip()

            if parsed_name in ApiTaskRunner._lora_lookup_cache:
                parsed_name = ApiTaskRunner._lora_lookup_cache[parsed_name]
            else:
                print(f"Warning: LoRA '{parsed_name}' wasn't found.")

            try:
                multiplier = float(match.group(2).strip())
            except ValueError:
                multiplier = 1.0

            extracted_loras.append({
                "path": parsed_name,
                "multiplier": multiplier,
                "is_high_noise": False
            })
            return ""

        cleaned_text = LORA_TAG_PATTERN.sub(match_handler, text)

        return cleaned_text, extracted_loras

    _EXTRA_ARGS_TAG_RE = re.compile(r"<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>")

    # Legacy SD-API-style sampler names -> raw sd.cpp sample_method names
    # accepted by SDGenerationParams::from_json_str (tag path has no alias map).
    # WebUI dropdown values are already raw names (parsed from the server
    # --help text); these aliases only cover old SD-API-style presets.
    _SAMPLER_TAG_ALIASES = {
        'euler a': 'euler_a',
        'euler': 'euler',
        'ddim': 'ddim_trailing',
        'res multistep': 'res_multistep',
        'res 2s': 'res_2s',
    }

    @classmethod
    def _to_tag_sampler(cls, name: str) -> str:
        n = str(name).strip().lower()
        if n.startswith('k_'):
            n = n[2:]
        return cls._SAMPLER_TAG_ALIASES.get(n, n)

    def _get_extra_args_string(self) -> str:
        """Build the <sd_cpp_extra_args> tag using the NESTED sdcpp schema
        accepted by SDGenerationParams::from_json_str (docs/api.md:540-575).
        The server silently skips unknown/flat keys, so only nested keys go
        in here."""
        def _num(p_key, cast, cond_key=None):
            if cond_key is not None and not self._get_param(cond_key, False):
                return None
            val = self._get_param(p_key)
            if val in (None, ''):
                return None
            try:
                return cast(val)
            except (ValueError, TypeError):
                return None

        sample_params = {}
        guidance = {}

        v = _num('in_steps', int)
        if v is not None:
            sample_params['sample_steps'] = v
        v = _num('in_cfg', float)
        if v is not None:
            guidance['txt_cfg'] = v
        v = _num('in_guidance', float, 'in_guidance_bool')
        if v is not None:
            guidance['distilled_guidance'] = v
        v = _num('in_flow_shift', float, 'in_flow_shift_bool')
        if v is not None:
            sample_params['flow_shift'] = v
        v = _num('in_eta', float, 'in_eta_bool')
        if v is not None:
            sample_params['eta'] = v
        v = _num('in_timestep_shift', int, 'in_timestep_shift_bool')
        if v is not None:
            sample_params['shifted_timestep'] = v

        val = self._get_param('in_sampling')
        if val is not None and str(val).strip() != "":
            sample_params['sample_method'] = self._to_tag_sampler(val)
        val = self._get_param('in_scheduler')
        if val is not None and str(val).strip() != "":
            s = str(val).strip().lower()
            sample_params['scheduler'] = 'discrete' if s == 'normal' else s

        if guidance:
            sample_params['guidance'] = guidance
        extra_args = {}
        if sample_params:
            extra_args['sample_params'] = sample_params

        v = _num('in_seed', int)
        if v is not None:
            extra_args['seed'] = v
        v = _num('in_clip_skip', int)
        if v is not None:
            extra_args['clip_skip'] = v
        v = _num('in_strength', float)
        if v is not None:
            extra_args['strength'] = v

        extra_args['preview'] = 'proj'
        extra_args['preview_interval'] = 1

        return (f"<sd_cpp_extra_args>{json.dumps(extra_args)}</sd_cpp_extra_args>"
                if extra_args else "")

    def _merge_extra_args(self, prompt: str, extra: dict) -> str:
        """Merge `extra` into an existing tag (the server only extracts the
        FIRST tag per prompt) or append a new one."""
        m = self._EXTRA_ARGS_TAG_RE.search(prompt or "")
        if m:
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                data = {}
            sp = data.setdefault('sample_params', {})
            for k, val in extra.get('sample_params', {}).items():
                if k == 'guidance':
                    sp.setdefault('guidance', {}).update(val)
                else:
                    sp[k] = val
            for k, val in extra.items():
                if k != 'sample_params':
                    data[k] = val
            tag = f"<sd_cpp_extra_args>{json.dumps(data)}</sd_cpp_extra_args>"
            return prompt[:m.start()] + tag + prompt[m.end():]
        return (prompt or "") + f" <sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"

    def _build_payload(self) -> dict:
        """Constructs the JSON payload for sdapi endpoints (blocking path,
        used by imgedit)."""
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
            "seed": self.get_seed(),
            "steps": int(self._get_param('in_steps', 20)),
            "cfg_scale": float(self._get_param('in_cfg', 7.0)),
        }

        if loras:
            unique_loras = {lora['path']: lora for lora in loras}.values()
            payload["lora"] = list(unique_loras)

        tag = self._get_extra_args_string()
        if tag:
            payload["prompt"] = payload["prompt"] + f" {tag}"

        return payload

    def _build_sdcpp_payload(self) -> dict:
        """Constructs the native sdcpp img_gen payload (async path)."""
        raw_pprompt = self._get_param('in_pprompt', '')
        raw_nprompt = self._get_param('in_nprompt', '')

        clean_pprompt, pprompt_loras = self._extract_loras(raw_pprompt)
        clean_nprompt, nprompt_loras = self._extract_loras(raw_nprompt)

        loras = pprompt_loras + nprompt_loras

        def _num(p_key, cast, cond_key=None, default=None):
            if cond_key is not None and not self._get_param(cond_key, False):
                return default
            val = self._get_param(p_key)
            if val in (None, ''):
                return default
            try:
                return cast(val)
            except (ValueError, TypeError):
                return default

        sample_params = {}
        guidance = {}

        v = _num('in_steps', int)
        if v is not None:
            sample_params['sample_steps'] = v
        v = _num('in_cfg', float)
        if v is not None:
            guidance['txt_cfg'] = v
        v = _num('in_guidance', float, 'in_guidance_bool')
        if v is not None:
            guidance['distilled_guidance'] = v
        v = _num('in_flow_shift', float, 'in_flow_shift_bool')
        if v is not None:
            sample_params['flow_shift'] = v
        v = _num('in_eta', float, 'in_eta_bool')
        if v is not None:
            sample_params['eta'] = v
        v = _num('in_timestep_shift', int, 'in_timestep_shift_bool')
        if v is not None:
            sample_params['shifted_timestep'] = v

        val = self._get_param('in_sampling')
        if val is not None and str(val).strip() != "":
            sample_params['sample_method'] = self._to_tag_sampler(val)
        val = self._get_param('in_scheduler')
        if val is not None and str(val).strip() != "":
            s = str(val).strip().lower()
            sample_params['scheduler'] = 'discrete' if s == 'normal' else s

        if guidance:
            sample_params['guidance'] = guidance

        payload = {
            "prompt": clean_pprompt,
            "negative_prompt": clean_nprompt,
            "width": int(self._get_param('in_width', 512)),
            "height": int(self._get_param('in_height', 512)),
            "batch_count": int(self._get_param('in_batch_count', 1)),
            "seed": self.get_seed(),
            "sample_params": sample_params,
            "preview": "proj",
            "preview_interval": 1,
            "output_format": "png",
        }

        v = _num('in_clip_skip', int)
        if v is not None:
            payload['clip_skip'] = v
        v = _num('in_strength', float)
        if v is not None:
            payload['strength'] = v

        if loras:
            unique_loras = {lora['path']: lora for lora in loras}.values()
            payload["lora"] = list(unique_loras)

        return payload

    def _process_response(self, data: dict):
        """Process the sdapi response (blocking path, used by imgedit)."""
        base, ext = os.path.splitext(self.output_path)
        begin_idx = max(0, self._int_param('in_output_begin_idx') or 0)
        images_list = data.get("images", []) or [
            item.get("b64_json")
            for item in data.get("data", [])
            if "b64_json" in item
        ]

        for i, b64_data in enumerate(images_list):
            if not b64_data:
                continue
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]

            image_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(image_bytes))
            if has_output_index_spec(self.output_path):
                target_path = expand_output_index(self.output_path, begin_idx + i)
            elif i == 0:
                target_path = self.output_path
            else:
                target_path = f"{base}_{begin_idx + i}{ext}"
            if ext.lower() == '.png':
                meta_string = build_a1111_metadata(
                    self.params, self.get_seed()
                )
                save_image_with_metadata(image, target_path, meta_string)
            else:
                image.save(target_path)
            self.outputs.append(target_path)

    def _process_sdcpp_response(self, result: dict):
        """Process the sdcpp job result (async path)."""
        base, ext = os.path.splitext(self.output_path)
        begin_idx = max(0, self._int_param('in_output_begin_idx') or 0)
        images_list = result.get("images", []) or []

        for i, img_data in enumerate(images_list):
            b64_data = img_data.get("b64_json")
            if not b64_data:
                continue
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]

            image_bytes = base64.b64decode(b64_data)
            image = Image.open(io.BytesIO(image_bytes))
            if has_output_index_spec(self.output_path):
                target_path = expand_output_index(self.output_path, begin_idx + i)
            elif i == 0:
                target_path = self.output_path
            else:
                target_path = f"{base}_{begin_idx + i}{ext}"
            if ext.lower() == '.png':
                meta_string = build_a1111_metadata(
                    self.params, self.get_seed()
                )
                save_image_with_metadata(image, target_path, meta_string)
            else:
                image.save(target_path)
            self.outputs.append(target_path)

    def run(self) -> Generator:
        self._resolve_paths()
        if self.use_async:
            yield from self._run_async()
        else:
            yield from self._run_blocking()

    def _run_async(self) -> Generator:
        """Submit to /sdcpp/v1/img_gen and poll job status."""
        payload = self._build_sdcpp_payload()
        self.fcommand = json.dumps(payload, indent=4)

        yield (
            self.fcommand,
            gr.update(visible=True, value=0),
            gr.update(visible=True, value="Submitting..."),
            gr.skip(),
            None
        )

        try:
            response = httpx.post(self.url, json=payload, timeout=30.0)
            if response.status_code not in (200, 201, 202):
                body = (response.text or "").strip()
                detail = f" {body[:200]}" if body else ""
                yield (
                    self.fcommand,
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(
                        value=f"Error: server returned HTTP "
                              f"{response.status_code}{detail}"
                    ),
                    None
                )
                return

            job_data = response.json()
            job_id = job_data.get("id")
            if not job_id:
                yield (
                    self.fcommand,
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(value="Error: no job id in response"),
                    None
                )
                return

            poll_url = f"http://{self.ip}:{self.port}/sdcpp/v1/jobs/{job_id}"

            prev_stats = server_state.last_generation_stats
            last_preview = None
            last_status = None

            while True:
                if not server_state.running:
                    return

                try:
                    r = httpx.get(poll_url, timeout=5.0)
                    if r.status_code != 200:
                        time.sleep(0.4)
                        continue

                    job = r.json()
                    status = job.get("status", "")

                    if status != last_status:
                        last_status = status
                        yield (
                            self.fcommand,
                            gr.update(visible=True, value=0),
                            gr.update(visible=True, value=f"{status}"),
                            gr.skip(),
                            gr.skip()
                        )

                    preview = job.get("preview")
                    if isinstance(preview, dict):
                        preview_b64 = preview.get("b64_json")
                        if preview_b64 and preview_b64 != last_preview:
                            last_preview = preview_b64
                            try:
                                preview_img = Image.open(
                                    io.BytesIO(base64.b64decode(preview_b64))
                                )
                                step = preview.get("step", 0) or 0
                                total = preview.get("total_steps", 1) or 1
                                pct = int(step / total * 100)
                                yield (
                                    self.fcommand,
                                    gr.update(visible=True, value=pct),
                                    gr.update(
                                        visible=True,
                                        value=f"Generating ({step}/{total})"
                                    ),
                                    gr.skip(),
                                    [preview_img]
                                )
                            except Exception:
                                pass

                    if status == "completed":
                        result = job.get("result") or {}
                        self._process_sdcpp_response(result)

                        gen_stats = server_state.last_generation_stats
                        if gen_stats == prev_stats and server_state.running:
                            deadline = time.time() + 1.5
                            while (
                                server_state.last_generation_stats == prev_stats
                                and time.time() < deadline
                            ):
                                time.sleep(0.1)
                            gen_stats = server_state.last_generation_stats
                        if not gen_stats:
                            gen_stats = "No stats recorded."

                        yield (
                            self.fcommand,
                            gr.update(visible=False, value=0),
                            gr.update(visible=False, value=""),
                            gr.update(value=gen_stats),
                            self.outputs
                        )
                        return

                    if status == "cancelled":
                        yield (
                            self.fcommand,
                            gr.update(visible=False, value=0),
                            gr.update(visible=False, value=""),
                            gr.update(value="Cancelled by server"),
                            None
                        )
                        return

                    if status == "failed":
                        error = job.get("error") or {}
                        msg = error.get("message", "unknown")
                        yield (
                            self.fcommand,
                            gr.update(visible=False, value=0),
                            gr.update(visible=False, value=""),
                            gr.update(value=f"Error: {msg}"),
                            None
                        )
                        return
                except Exception:
                    pass

                time.sleep(0.4)
        except Exception as exc:
            ApiTaskRunner._lora_lookup_cache.clear()
            yield (
                self.fcommand,
                gr.update(visible=False, value=0),
                gr.update(visible=False, value=""),
                gr.update(value=f"Error: {type(exc).__name__}: {exc}"),
                None
            )

    def _run_blocking(self) -> Generator:
        """Synchronous path (used by imgedit)."""
        payload_or_files = self._build_payload()

        if isinstance(payload_or_files, tuple):
            self.fcommand = json.dumps(payload_or_files[0], indent=4)
        else:
            display_payload = payload_or_files.copy()
            for key in ["init_images", "mask"]:
                if key in display_payload:
                    if isinstance(display_payload[key], list):
                        display_payload[key] = [
                            img[:60] + "..." for img in display_payload[key]
                        ]
                    elif isinstance(display_payload[key], str):
                        display_payload[key] = (
                            display_payload[key][:60] + "..."
                        )
            self.fcommand = json.dumps(display_payload, indent=4)

        yield (
            self.fcommand,
            gr.update(visible=True, value=0),
            gr.update(visible=True, value="Starting..."),
            gr.skip(),
            None
        )

        prev_stats = server_state.last_generation_stats

        try:
            result_holder = {}

            def _do_request():
                try:
                    if isinstance(payload_or_files, tuple):
                        data, files = payload_or_files
                        result_holder['response'] = httpx.post(
                            self.url, data=data, files=files, timeout=None
                        )
                    else:
                        result_holder['response'] = httpx.post(
                            self.url, json=payload_or_files, timeout=None
                        )
                except Exception as exc:
                    result_holder['error'] = exc

            worker = threading.Thread(target=_do_request, daemon=True)
            worker.start()

            progress_url = f"http://{self.ip}:{self.port}/sdapi/v1/progress"
            last_preview = None
            last_pct = -1
            while worker.is_alive():
                if not server_state.running:
                    return

                try:
                    r = httpx.get(progress_url, timeout=1.0)
                    if r.status_code == 200:
                        pdata = r.json()
                        pct = int(round(pdata.get("progress", 0.0) * 100))
                        img_b64 = pdata.get("current_image")

                        img_update = gr.skip()
                        changed = False

                        if img_b64 and img_b64 != last_preview:
                            last_preview = img_b64
                            img_update = [
                                Image.open(
                                    io.BytesIO(base64.b64decode(img_b64))
                                )
                            ]
                            changed = True

                        if pct != last_pct:
                            last_pct = pct
                            changed = True

                        if changed:
                            yield (
                                self.fcommand,
                                gr.update(visible=True, value=last_pct),
                                gr.update(
                                    visible=True,
                                    value=f"Generating ({last_pct}%)"
                                ),
                                gr.skip(),
                                img_update
                            )
                except Exception:
                    pass
                worker.join(timeout=0.4)
            worker.join()

            if 'error' in result_holder:
                raise result_holder['error']
            response = result_holder['response']

            if response.status_code == 200:
                self._process_response(response.json())
                gen_stats = server_state.last_generation_stats
                if gen_stats == prev_stats and server_state.running:
                    deadline = time.time() + 1.5
                    while (
                        server_state.last_generation_stats == prev_stats
                        and time.time() < deadline
                    ):
                        time.sleep(0.1)
                    gen_stats = server_state.last_generation_stats
                if not gen_stats:
                    gen_stats = "No stats recorded."

                yield (
                    self.fcommand,
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(value=gen_stats),
                    self.outputs
                )
            else:
                ApiTaskRunner._lora_lookup_cache.clear()
                body = (response.text or "").strip()
                detail = f" {body[:200]}" if body else ""
                yield (
                    self.fcommand,
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(
                        value=f"Error: server returned HTTP "
                              f"{response.status_code}{detail}"
                    ),
                    None
                )
        except Exception as exc:
            ApiTaskRunner._lora_lookup_cache.clear()
            yield (
                self.fcommand,
                gr.update(visible=False, value=0),
                gr.update(visible=False, value=""),
                gr.update(value=f"Error: {type(exc).__name__}: {exc}"),
                None
            )


class Txt2ImgApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(
            dir_key='txt2img_dir', subctrl_id=0, extension='png'
        )
        self.use_async = True
        self.url = f"http://{self.ip}:{self.port}/sdcpp/v1/img_gen"


class Img2ImgApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(
            dir_key='img2img_dir', subctrl_id=1, extension='png'
        )
        self.use_async = True
        self.url = f"http://{self.ip}:{self.port}/sdcpp/v1/img_gen"

    def _build_sdcpp_payload(self) -> dict:
        payload = super()._build_sdcpp_payload()

        init_img = image_to_pil(
            self._get_param('in_img_inp') or self._get_param('in_first_frame_inp')
        )
        if init_img is not None:
            buf = io.BytesIO()
            init_img.save(buf, format="PNG")
            payload["init_image"] = base64.b64encode(
                buf.getvalue()
            ).decode('utf-8')

        mask_input = self._get_param('in_img_mask') or self._get_param('in_mask_img')
        mask_img = process_editor_mask(mask_input)

        if mask_img is not None:
            if self._get_param('in_invert_mask', False):
                mask_img = ImageOps.invert(mask_img.convert('L'))
            m_buf = io.BytesIO()
            mask_img.save(m_buf, format="PNG")
            payload["mask_image"] = base64.b64encode(
                m_buf.getvalue()
            ).decode('utf-8')

        if self._get_param('in_img_cfg_bool'):
            try:
                img_cfg = float(self._get_param('in_img_cfg'))
                payload.setdefault('sample_params', {}) \
                       .setdefault('guidance', {})['img_cfg'] = img_cfg
            except (TypeError, ValueError):
                pass

        return payload

    def _build_payload(self) -> dict:
        # Dead code for the async path, kept for API symmetry if use_async
        # is ever toggled off.
        payload = super()._build_payload()
        init_img = image_to_pil(
            self._get_param('in_img_inp') or self._get_param('in_first_frame_inp')
        )
        if init_img is not None:
            buf = io.BytesIO()
            init_img.save(buf, format="PNG")
            payload["init_images"] = [
                base64.b64encode(buf.getvalue()).decode('utf-8')
            ]

        mask_input = self._get_param('in_img_mask') or self._get_param('in_mask_img')
        mask_img = process_editor_mask(mask_input)

        if mask_img is not None:
            m_buf = io.BytesIO()
            mask_img.save(m_buf, format="PNG")
            payload["mask"] = base64.b64encode(m_buf.getvalue()).decode('utf-8')
            payload["inpainting_mask_invert"] = self._get_param(
                'in_invert_mask', False
            )

        strength = self._get_param('in_strength', 0.75)
        if strength not in (None, ''):
            payload["denoising_strength"] = float(strength)

        if self._get_param('in_img_cfg_bool'):
            try:
                img_cfg = float(self._get_param('in_img_cfg'))
            except (TypeError, ValueError):
                img_cfg = None
            if img_cfg is not None:
                payload["prompt"] = self._merge_extra_args(
                    payload["prompt"],
                    {"sample_params": {"guidance": {"img_cfg": img_cfg}}},
                )
        return payload


class ImgEditApiRunner(ApiTaskRunner):
    def prepare(self):
        self._set_output_path(dir_key='imgedit_dir', subctrl_id=2, extension='png')
        self.use_async = False
        self.url = f"http://{self.ip}:{self.port}/v1/images/edits"

    def _build_payload(self) -> tuple:
        """Constructs multipart form data and files."""
        extra_args_str = self._get_extra_args_string()
        prompt = self._get_param('in_pprompt', '') + extra_args_str

        form_data = {
            "prompt": prompt,
            "n": str(self._int_param('in_batch_count', 1)),
            "size": (f"{self._int_param('in_width', 512)}x"
                    f"{self._int_param('in_height', 512)}"),
            "output_format": "png"
        }

        files = []
        init_imgs = self._get_param('in_ref_img') or self._get_param('in_img_inp')
        if init_imgs is not None:
            if not isinstance(init_imgs, list):
                init_imgs = [init_imgs]

            for idx, img_item in enumerate(init_imgs):
                img_obj = image_to_pil(img_item)
                if img_obj is None:
                    continue

                buf = io.BytesIO()
                img_obj.save(buf, format="PNG")
                buf.seek(0)

                files.append(
                    ("image[]", (f"image_{idx}.png", buf, "image/png"))
                )

        mask_input = self._get_param('in_img_mask') or self._get_param('in_mask_img')
        mask_img = process_editor_mask(mask_input)

        if mask_img is not None:
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

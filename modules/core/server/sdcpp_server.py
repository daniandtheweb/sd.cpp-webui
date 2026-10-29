"""sd.cpp-webui - core - stable-diffusion.cpp server"""

import os
import io
import json
import time
import base64
import threading
from typing import Dict, Any, Generator, Optional

import httpx
from PIL import Image, ImageOps, PngImagePlugin

import gradio as gr

from modules.core.common.sd_common import (
    CommonRunner, LORA_TAG_PATTERN, image_to_pil, process_editor_mask,
    has_output_index_spec, expand_output_index
)
from modules.loader import get_loras
from modules.utils.metadata_utils import (
    find_sdcpp_generation, SDCPP_SEPARATOR
)
from modules.shared_instance import server_state


HIGH_NOISE_LORA_PREFIX = "|high_noise|"

# Base64 fields shortened in the displayed payload
_IMAGE_FIELDS = (
    "init_image", "end_image", "mask_image", "control_image",
    "ip_adapter_image", "ref_images", "control_frames", "image",
)


def encode_image(img: Image.Image) -> str:
    """Encodes a PIL image as a base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def decode_bytes(b64_data: str) -> bytes:
    """Decodes a base64 string or data URL."""
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    return base64.b64decode(b64_data)


def decode_image(b64_data: str) -> Image.Image:
    """Decodes a base64 string or data URL into a PIL image."""
    return Image.open(io.BytesIO(decode_bytes(b64_data)))


def _flatten_postprocess(operation: Any) -> list:
    """
    Flattens the upscale record (first pass + its 'postprocess' list)
    into a list of passes.
    """
    if isinstance(operation, str):
        try:
            operation = json.loads(operation)
        except json.JSONDecodeError:
            return []
    if not isinstance(operation, dict):
        return []
    first = dict(operation)
    later = first.pop("postprocess", None) or []
    return [first] + [op for op in later if isinstance(op, dict)]


def merge_upscale_metadata(source: bytes, upscaled: bytes,
                           operation: Any) -> bytes:
    """
    Carries the source metadata over to the upscaled image.

    The upscale endpoint only records the upscale itself, so the
    passes are appended to the source record's 'postprocess' list,
    like sd-cli does with --upscale-model.
    """
    source_text = Image.open(io.BytesIO(source)).info.get("parameters")
    image = Image.open(io.BytesIO(upscaled))

    # No source metadata (--disable-image-metadata): keep it that way
    pnginfo = None
    if source_text:
        text = source_text
        found = find_sdcpp_generation(source_text)
        if found:
            infotext, record = found
            record = dict(record)
            record["postprocess"] = (
                list(record.get("postprocess") or [])
                + _flatten_postprocess(operation)
            )
            text = json.dumps(record, separators=(",", ":"),
                              ensure_ascii=False)
            if infotext:
                text = f"{infotext}, {SDCPP_SEPARATOR}{text}"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", text)

    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


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


def redact_payload(payload: dict) -> dict:
    """Returns a copy of the payload with the base64 images shortened."""
    shown = dict(payload)
    for key in _IMAGE_FIELDS:
        val = shown.get(key)
        if isinstance(val, str) and len(val) > 60:
            shown[key] = f"<base64 {len(val)} bytes>"
        elif isinstance(val, list):
            shown[key] = [
                f"<base64 {len(v)} bytes>" if isinstance(v, str) else v
                for v in val
            ]
    return shown


class _ActiveJob:
    """Tracks the job running on the server so it can be cancelled."""

    def __init__(self):
        self._lock = threading.Lock()
        self.base_url = None
        self.job_id = None
        self.cancel_requested = threading.Event()

    def start(self, base_url: str):
        with self._lock:
            self.base_url = base_url
            self.job_id = None
            self.cancel_requested.clear()

    def set_id(self, job_id: str):
        with self._lock:
            self.job_id = job_id

    def finish(self):
        with self._lock:
            self.base_url = None
            self.job_id = None

    def cancel(self) -> bool:
        """Requests the cancellation of the active job."""
        with self._lock:
            base_url, job_id = self.base_url, self.job_id
            if base_url is None:
                return False
            self.cancel_requested.set()
        # Not submitted yet, the runner checks the flag before submitting
        if job_id is None:
            return True
        try:
            httpx.post(
                f"{base_url}/sdcpp/v1/jobs/{job_id}/cancel", timeout=5.0
            )
        except Exception as exc:
            print(f"Error cancelling job {job_id}: {exc}")
        return True


active_job = _ActiveJob()


def cancel_active_job():
    """Cancels the generation running on the server, if any."""
    active_job.cancel()


class ApiTaskRunner(CommonRunner):
    """
    Builds and manages API requests to the sd.cpp server.
    """

    endpoint = "/sdcpp/v1/img_gen"
    dir_key = "txt2img_dir"
    subctrl_id = 0
    extension = "png"

    POLL_INTERVAL = 0.4

    _lora_lookup_cache = {}

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # The server may have been started from another tab
        if server_state.running and server_state.ip:
            self.ip = str(server_state.ip)
            self.port = str(server_state.port)
        else:
            self.ip = str(self._get_param('in_ip'))
            self.port = str(self._int_param('in_port'))
        self.base_url = f"http://{self.ip}:{self.port}"
        self.url = ""

        self.output_path = ""
        self.outputs = []
        self.notes = []

    def prepare(self):
        """Sets the output path and the endpoint."""
        self._set_output_path(
            dir_key=self.dir_key, subctrl_id=self.subctrl_id,
            extension=self.extension
        )
        self.url = f"{self.base_url}{self.endpoint}"

    def _num(self, key: str, cast, cond_key: str = None):
        """
        Helper to get a numeric parameter.
        Returns None if empty or if its checkbox (cond_key) is disabled.
        """
        if cond_key is not None and not self._get_param(cond_key, False):
            return None
        val = self._get_param(key)
        if val in (None, ''):
            return None
        try:
            # Gradio Number/Slider widgets yield floats (e.g. 512.0)
            return cast(float(val)) if cast is int else cast(val)
        except (ValueError, TypeError):
            return None

    def _text(self, key: str) -> Optional[str]:
        """Helper to get a stripped string parameter, None if empty."""
        val = self._get_param(key)
        if val is None:
            return None
        val = str(val).strip()
        return val or None

    @staticmethod
    def _model_stem(name: Any) -> Optional[str]:
        """The server looks up upscalers by file stem."""
        if not name:
            return None
        return os.path.splitext(os.path.basename(str(name)))[0] or None

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

        def _strip_prefix(name):
            # <lora:|high_noise|name:multiplier> applies to the high noise model
            name = name.strip()
            if name.startswith(HIGH_NOISE_LORA_PREFIX):
                return name[len(HIGH_NOISE_LORA_PREFIX):].strip(), True
            return name, False

        if any(_strip_prefix(name)[0] not in ApiTaskRunner._lora_lookup_cache
               for name, _ in matches):
            ApiTaskRunner._lora_lookup_cache.clear()
            for lora_path in get_loras():
                base_name = os.path.splitext(os.path.basename(lora_path))[0]
                path_no_ext = os.path.splitext(lora_path)[0]
                base_with_ext = os.path.basename(lora_path)
                for key in (base_name, path_no_ext, base_with_ext, lora_path):
                    ApiTaskRunner._lora_lookup_cache[key] = lora_path

        def match_handler(match):
            parsed_name, is_high_noise = _strip_prefix(match.group(1))

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
                "is_high_noise": is_high_noise
            })
            return ""

        cleaned_text = LORA_TAG_PATTERN.sub(match_handler, text)
        return cleaned_text.strip(), extracted_loras

    # Legacy SD-API-style sampler names -> raw sd.cpp sample_method names
    _SAMPLER_ALIASES = {
        'euler a': 'euler_a',
        'euler': 'euler',
        'ddim': 'ddim_trailing',
        'res multistep': 'res_multistep',
        'res 2s': 'res_2s',
    }

    @classmethod
    def _to_api_sampler(cls, name: str) -> str:
        n = str(name).strip().lower()
        if n.startswith('k_'):
            n = n[2:]
        return cls._SAMPLER_ALIASES.get(n, n)

    def _sample_params(self) -> dict:
        """Builds the sample_params object."""
        sample_params = {}
        guidance = {}

        # Steps
        v = self._num('in_steps', int)
        if v is not None:
            sample_params['sample_steps'] = v
        # Flow Shift
        v = self._num('in_flow_shift', float, 'in_flow_shift_bool')
        if v is not None:
            sample_params['flow_shift'] = v
        # ETA for DDIM and TCD
        v = self._num('in_eta', float, 'in_eta_bool')
        if v is not None:
            sample_params['eta'] = v
        # Timestep shift for NitroFusion
        v = self._num('in_timestep_shift', int, 'in_timestep_shift_bool')
        if v is not None:
            sample_params['shifted_timestep'] = v

        # Sampler & Scheduler
        sampler = self._text('in_sampling')
        if sampler:
            sample_params['sample_method'] = self._to_api_sampler(sampler)
        scheduler = self._text('in_scheduler')
        if scheduler:
            s = scheduler.lower()
            sample_params['scheduler'] = 'discrete' if s == 'normal' else s

        # Sigmas
        sigmas = parse_number_list(self._get_param('in_sigmas'))
        if sigmas:
            sample_params['custom_sigmas'] = sigmas

        # Guidance
        v = self._num('in_cfg', float)
        if v is not None:
            guidance['txt_cfg'] = v
        v = self._num('in_img_cfg', float, 'in_img_cfg_bool')
        if v is not None:
            guidance['img_cfg'] = v
        v = self._num('in_guidance', float, 'in_guidance_bool')
        if v is not None:
            guidance['distilled_guidance'] = v

        # Skip Layer Guidance (SLG)
        if self._get_param('in_slg_bool'):
            slg = {}
            v = self._num('in_slg_scale', float)
            if v is not None:
                slg['scale'] = v
            v = self._num('in_skip_layer_start', float)
            if v is not None:
                slg['layer_start'] = v
            v = self._num('in_skip_layer_end', float)
            if v is not None:
                slg['layer_end'] = v
            layers = parse_number_list(self._get_param('in_skip_layers'), int)
            if layers:
                slg['layers'] = layers
            if slg:
                guidance['slg'] = slg

        if guidance:
            sample_params['guidance'] = guidance
        return sample_params

    def _hires(self) -> Optional[dict]:
        """Builds the hires object."""
        # Only if the tab has the HiRes Fix settings
        if 'in_hires_bool' not in self.params:
            return None
        if not self._get_param('in_hires_bool'):
            return {"enabled": False}
        hires = {
            "enabled": True,
            "upscaler": self._model_stem(
                self._get_param('in_hires_upscaler') or 'Latent'
            ),
        }
        for key, field, cast in (
            ('in_hires_scale', 'scale', float),
            ('in_hires_width', 'target_width', int),
            ('in_hires_height', 'target_height', int),
            ('in_hires_steps', 'steps', int),
            ('in_hires_denoising_strength', 'denoising_strength', float),
            ('in_hires_upscale_tile_size', 'upscale_tile_size', int),
        ):
            v = self._num(key, cast)
            if v is not None:
                hires[field] = v
        sigmas = parse_number_list(self._get_param('in_hires_sigmas'))
        if sigmas:
            hires['custom_sigmas'] = sigmas
        return hires

    def _vae_tiling(self) -> Optional[dict]:
        """Builds the vae_tiling_params object."""
        # Only if the tab has the VAE Tiling settings
        if 'in_vae_tiling' not in self.params:
            return None
        tiling = {"enabled": bool(self._get_param('in_vae_tiling'))}
        # LTX Specifics
        if self._get_param('in_temporal_tiling'):
            tiling['temporal_tiling'] = True
        if not tiling['enabled']:
            return tiling
        v = self._num('in_vae_tile_overlap', float)
        if v is not None:
            tiling['target_overlap'] = v
        size = self._num('in_vae_tile_size', int)
        if size:
            tiling['tile_size_w'] = tiling['tile_size_h'] = size
        if self._get_param('in_vae_relative_bool'):
            rel = self._num('in_vae_relative_tile_size', float)
            if rel:
                tiling['rel_size_w'] = tiling['rel_size_h'] = rel
        return tiling

    def _cache(self) -> dict:
        """Builds the cache fields."""
        # Only if the tab has the Cache settings
        if 'in_cache_bool' not in self.params:
            return {}
        if not self._get_param('in_cache_bool'):
            return {"cache_mode": "disabled"}
        cache = {}
        mode = self._text('in_cache_mode')
        if mode:
            cache['cache_mode'] = mode
        option = self._text('in_cache_option')
        if option and option.strip('"'):
            cache['cache_option'] = option.strip('"')
        mask = self._text('in_scm_mask')
        if mask:
            cache['scm_mask'] = mask
        policy = self._text('in_scm_policy')
        if policy in ('dynamic', 'static'):
            cache['scm_policy_dynamic'] = policy == 'dynamic'
        return cache

    def _preview(self) -> tuple:
        """Returns (preview_mode, preview_interval)."""
        # Tabs without the Preview settings use the projection preview
        if 'in_preview_bool' not in self.params:
            return "proj", 1
        if not self._get_param('in_preview_bool'):
            return "none", 1
        mode = self._text('in_preview_mode') or "proj"
        interval = self._num('in_preview_interval', int) or 1
        return mode, max(1, interval)

    def _ref_image_args(self) -> Optional[str]:
        """
        Builds the ref_image_args key=value list from the reference
        image settings.
        Returns None when nothing is set (model preset is auto-detected).
        """
        args = []
        if self._get_param('in_increase_ref_index'):
            args.append("ref_index_mode=increase")
        if self._get_param('in_disable_auto_resize_ref_image'):
            args.append("resize_before_vae=false")
        return ",".join(args) if args else None

    def _control(self) -> dict:
        """Builds the ControlNet fields."""
        if not self._get_param('in_cnnet_bool'):
            return {}
        control_img = image_to_pil(self._get_param('in_control_img'))
        if control_img is None:
            return {}
        control = {"control_image": encode_image(control_img.convert('RGB'))}
        v = self._num('in_control_strength', float)
        if v is not None:
            control['control_strength'] = v
        return control

    def _add_input_images(self, payload: dict):
        """
        Adds the input images to the payload.
        To be implemented by subclasses.
        """

    def _build_payload(self) -> dict:
        """Constructs the JSON payload for /sdcpp/v1/img_gen."""
        clean_pprompt, pprompt_loras = self._extract_loras(
            self._get_param('in_pprompt', '') or ''
        )
        clean_nprompt, nprompt_loras = self._extract_loras(
            self._get_param('in_nprompt', '') or ''
        )

        preview_mode, preview_interval = self._preview()

        payload = {
            "prompt": clean_pprompt,
            "negative_prompt": clean_nprompt,
            "width": self._int_param('in_width', 512),
            "height": self._int_param('in_height', 512),
            "batch_count": self._int_param('in_batch_count', 1) or 1,
            "seed": self.get_seed(),
            "sample_params": self._sample_params(),
            "preview": preview_mode,
            "preview_interval": preview_interval,
            "output_format": "png",
        }

        v = self._num('in_clip_skip', int)
        if v is not None:
            payload['clip_skip'] = v

        # Output
        if 'in_disable_img_metadata' in self.params:
            payload['embed_image_metadata'] = not self._get_param(
                'in_disable_img_metadata'
            )
        v = self._num('in_compression_quality', int)
        if v is not None:
            payload['output_compression'] = v

        # LoRA
        loras = pprompt_loras + nprompt_loras
        if loras:
            unique = {(l['path'], l['is_high_noise']): l for l in loras}
            payload['lora'] = list(unique.values())

        # HiRes Fix
        hires = self._hires()
        if hires is not None:
            payload['hires'] = hires

        # VAE Tiling
        tiling = self._vae_tiling()
        if tiling is not None:
            payload['vae_tiling_params'] = tiling

        # Cache
        payload.update(self._cache())

        # Image Preprocessing
        preprocess = self._text('in_img_preprocess')
        if preprocess:
            payload['image_preprocess'] = preprocess

        # Reference images
        ref_args = self._ref_image_args()
        if ref_args:
            payload['ref_image_args'] = ref_args

        # ControlNet
        payload.update(self._control())

        self._add_input_images(payload)
        return payload

    def _target_path(self, i: int) -> str:
        """
        Output path of the i-th image. Mirrors sd-cli: with a %d/%Nd
        specifier in the path, the path expanded with the batch index
        (offset by the output begin index), otherwise <base>_<begin+i>.
        """
        base, ext = os.path.splitext(self.output_path)
        begin_idx = max(0, self._int_param('in_output_begin_idx') or 0)
        if has_output_index_spec(self.output_path):
            return expand_output_index(self.output_path, begin_idx + i)
        if i == 0:
            return self.output_path
        return f"{base}_{begin_idx + i}{ext}"

    def _post_upscale(self, data: bytes) -> bytes:
        """
        Upscales an image through /sdcpp/v1/upscale.
        Returns the original image if upscaling is disabled or fails.
        """
        if not self._get_param('in_upscl_bool'):
            return data
        upscaler = self._model_stem(self._get_param('in_upscl'))
        if not upscaler:
            return data
        body = {
            "image": base64.b64encode(data).decode('utf-8'),
            "upscaler": upscaler,
            "repeats": max(1, min(4, self._int_param('in_upscl_rep', 1) or 1)),
            "output_format": "png",
        }
        tile = self._int_param('in_upscl_tile_size')
        if tile:
            body['tile_size'] = tile
        try:
            resp = httpx.post(
                f"{self.base_url}/sdcpp/v1/upscale", json=body, timeout=None
            )
        except Exception as exc:
            self.notes.append(f"Upscale failed: {exc}")
            return data
        if resp.status_code != 200:
            detail = (resp.text or "").strip()[:200]
            self.notes.append(
                f"Upscale failed (HTTP {resp.status_code}): {detail}"
            )
            return data
        out = resp.json()
        images = out.get("images") or []
        if not images or not images[0].get("b64_json"):
            self.notes.append("Upscale returned no image")
            return data
        return merge_upscale_metadata(
            data, decode_bytes(images[0]["b64_json"]),
            out.get("generation_info")
        )

    def _save_result(self, result: dict):
        """
        Saves the images of a completed job.
        The server embeds the metadata, so the images are saved as-is.
        """
        for i, item in enumerate(result.get("images") or []):
            b64_data = item.get("b64_json")
            if not b64_data:
                continue
            data = self._post_upscale(decode_bytes(b64_data))
            target_path = self._target_path(i)
            with open(target_path, 'wb') as f:
                f.write(data)
            self.outputs.append(target_path)

    def _update(self, pct=0, status="", gallery=None):
        """Builds a progress update."""
        return (
            self.fcommand,
            gr.update(visible=True, value=pct),
            gr.update(visible=True, value=status),
            gr.skip(),
            gr.skip() if gallery is None else gallery,
        )

    def _final(self, stats: str, gallery=None):
        """Builds the final update."""
        return (
            self.fcommand,
            gr.update(visible=False, value=0),
            gr.update(visible=False, value=""),
            gr.update(value=stats),
            gallery,
        )

    def _wait_for_log_stats(self, prev_stats: str) -> str:
        """Waits briefly for the log parser to record the server stats."""
        gen_stats = server_state.last_generation_stats
        if gen_stats == prev_stats and server_state.running:
            deadline = time.time() + 1.5
            while (server_state.last_generation_stats == prev_stats
                   and time.time() < deadline):
                time.sleep(0.1)
            gen_stats = server_state.last_generation_stats
        return gen_stats if gen_stats != prev_stats else ""

    def _job_stats(self, job: dict, prev_stats: str) -> str:
        """Builds the statistics string of a completed job."""
        parts = []
        # Timestamps are in milliseconds
        started, completed = job.get("started"), job.get("completed")
        if isinstance(started, (int, float)) and isinstance(
                completed, (int, float)):
            parts.append(f"Job time: {(completed - started) / 1000:.2f}s")
        log_stats = self._wait_for_log_stats(prev_stats)
        if log_stats:
            parts.append(log_stats)
        parts.extend(self.notes)
        return " | ".join(parts) or "No stats recorded."

    def run(self) -> Generator:
        """Submits the job and yields Gradio updates."""
        payload = self._build_payload()
        self.fcommand = json.dumps(redact_payload(payload), indent=4)

        yield (
            self.fcommand,
            gr.update(visible=True, value=0),
            gr.update(visible=True, value="Submitting..."),
            gr.update(value=""),
            None,
        )

        active_job.start(self.base_url)
        try:
            yield from self._submit_and_poll(payload)
        except Exception as exc:
            ApiTaskRunner._lora_lookup_cache.clear()
            yield self._final(f"Error: {type(exc).__name__}: {exc}")
        finally:
            active_job.finish()

    def _submit_and_poll(self, payload: dict) -> Generator:
        """Submits to /sdcpp/v1/img_gen and polls the job status."""
        if active_job.cancel_requested.is_set():
            yield self._final("Cancelled")
            return

        # Drop the progress of the previous job
        server_state.latest_update = {}

        response = httpx.post(self.url, json=payload, timeout=30.0)
        if response.status_code not in (200, 201, 202):
            ApiTaskRunner._lora_lookup_cache.clear()
            body = (response.text or "").strip()
            detail = f" {body[:300]}" if body else ""
            yield self._final(
                f"Error: server returned HTTP {response.status_code}{detail}"
            )
            return

        job_id = response.json().get("id")
        if not job_id:
            yield self._final("Error: no job id in response")
            return
        active_job.set_id(job_id)
        # Stop was pressed while submitting
        if active_job.cancel_requested.is_set():
            active_job.cancel()

        poll_url = f"{self.base_url}/sdcpp/v1/jobs/{job_id}"
        prev_stats = server_state.last_generation_stats
        last_preview = None
        last_progress = None
        last_shown = None

        while True:
            if not server_state.running:
                yield self._final("Server stopped")
                return

            try:
                r = httpx.get(poll_url, timeout=5.0)
            except httpx.HTTPError:
                time.sleep(self.POLL_INTERVAL)
                continue
            if r.status_code in (404, 410):
                yield self._final(
                    f"Error: job {job_id} is no longer available"
                )
                return
            if r.status_code != 200:
                time.sleep(self.POLL_INTERVAL)
                continue

            job = r.json()
            status = job.get("status", "")

            if status == "completed":
                self._save_result(job.get("result") or {})
                yield self._final(
                    self._job_stats(job, prev_stats), self.outputs or None
                )
                return

            if status == "cancelled":
                yield self._final("Cancelled")
                return

            if status == "failed":
                error = job.get("error") or {}
                yield self._final(
                    f"Error: {error.get('message', 'unknown')}"
                )
                return

            # Preview
            gallery = None
            preview = job.get("preview")
            if isinstance(preview, dict):
                preview_b64 = preview.get("b64_json")
                if preview_b64 and preview_b64 != last_preview:
                    last_preview = preview_b64
                    try:
                        gallery = [decode_image(preview_b64)]
                    except Exception:
                        gallery = None

            # Progress: the server log is parsed like the CLI one
            # (Speed | ETA), the preview step is the fallback
            latest = server_state.latest_update
            if status == "queued":
                pos = job.get("queue_position")
                shown = (0, f"Queued (position {pos})" if pos else "Queued")
            elif isinstance(latest, dict) and 'percent' in latest:
                last_progress = latest
                shown = (latest['percent'], latest.get('status', ''))
            elif last_progress is not None:
                shown = (last_progress['percent'],
                         last_progress.get('status', ''))
            elif isinstance(preview, dict) and preview.get('total_steps'):
                step = preview.get('step', 0) or 0
                total = preview['total_steps']
                shown = (int(step / total * 100), f"Step: {step}/{total}")
            else:
                shown = (0, "Starting...")
            if active_job.cancel_requested.is_set():
                shown = (shown[0], "Cancelling...")

            if shown != last_shown or gallery is not None:
                last_shown = shown
                yield self._update(*shown, gallery=gallery)

            time.sleep(self.POLL_INTERVAL)


class Txt2ImgApiRunner(ApiTaskRunner):
    """Builds the txt2img request."""
    dir_key = "txt2img_dir"
    subctrl_id = 0


class Img2ImgApiRunner(ApiTaskRunner):
    """Builds the img2img request."""
    dir_key = "img2img_dir"
    subctrl_id = 1

    def _add_input_images(self, payload: dict):
        # Add img2img specific fields
        init_img = image_to_pil(
            self._get_param('in_img_inp')
            or self._get_param('in_first_frame_inp')
        )
        if init_img is not None:
            payload["init_image"] = encode_image(init_img)

        mask_input = (self._get_param('in_img_mask')
                      or self._get_param('in_mask_img'))
        mask_img = process_editor_mask(mask_input)
        if mask_img is not None:
            mask_img = mask_img.convert('L')
            if self._get_param('in_invert_mask', False):
                mask_img = ImageOps.invert(mask_img)
            payload["mask_image"] = encode_image(mask_img)

        v = self._num('in_strength', float)
        if v is not None:
            payload['strength'] = v


class ImgEditApiRunner(ApiTaskRunner):
    """Builds the image editing (instruct) request."""
    dir_key = "imgedit_dir"
    subctrl_id = 2

    def _add_input_images(self, payload: dict):
        ref_imgs = self._get_param('in_ref_img')
        if ref_imgs is None:
            ref_imgs = self._get_param('in_img_inp')
        if ref_imgs is None:
            return
        if not isinstance(ref_imgs, list):
            ref_imgs = [ref_imgs]

        encoded = []
        for item in ref_imgs:
            img = image_to_pil(item)
            if img is not None:
                encoded.append(encode_image(img))
        if encoded:
            payload["ref_images"] = encoded

        mask_input = (self._get_param('in_img_mask')
                      or self._get_param('in_mask_img'))
        mask_img = process_editor_mask(mask_input)
        if mask_img is not None:
            payload["mask_image"] = encode_image(mask_img.convert('L'))


def txt2img_api(params: dict):
    """Creates and runs a Txt2ImgApiRunner."""
    runner = Txt2ImgApiRunner(params)
    runner.prepare()
    yield from runner.run()


def img2img_api(params: dict):
    """Creates and runs an Img2ImgApiRunner."""
    runner = Img2ImgApiRunner(params)
    runner.prepare()
    yield from runner.run()


def imgedit_api(params: dict):
    """Creates and runs an ImgEditApiRunner."""
    runner = ImgEditApiRunner(params)
    runner.prepare()
    yield from runner.run()

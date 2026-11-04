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
from modules.ui.progress import progress_update
from modules.options import emit
from modules.options.convert import stem
from modules.utils.math_utils import remember_seed


HIGH_NOISE_LORA_PREFIX = "|high_noise|"

# Base64 fields shortened in the displayed payload
_IMAGE_FIELDS = (
    "init_image", "end_image", "mask_image", "control_image",
    "ip_adapter_image", "ref_images", "control_frames", "ref_audios",
    "image",
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

    # The job (and its preview image) is polled every POLL_INTERVAL,
    # the progress parsed from the server log every PROGRESS_INTERVAL
    POLL_INTERVAL = 0.4
    PROGRESS_INTERVAL = 0.1

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

    @staticmethod
    def _model_stem(name: Any) -> Optional[str]:
        """The server looks up upscalers by file stem."""
        return stem(name, None)

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

    def _encoded_image(self, key: str) -> Optional[str]:
        """The base64 PNG of an input image setting, None without one."""
        img = image_to_pil(self._get_param(key))
        if img is None:
            return None
        return encode_image(img.convert('RGB'))

    def _input_images(self) -> Dict[str, Any]:
        """
        The input images of the request (init_image, mask_image,
        ref_images). To be implemented by subclasses.
        """
        return {}

    def _build_payload(self) -> dict:
        """
        Constructs the JSON payload for /sdcpp/v1/img_gen. Where each
        field goes and when comes from modules/options/registry.py; the
        runner only computes what the registry can't: the prompts without
        their LoRA tags, the LoRAs and the encoded images.
        """
        clean_pprompt, pprompt_loras = self._extract_loras(
            self._get_param('in_pprompt', '') or ''
        )
        clean_nprompt, nprompt_loras = self._extract_loras(
            self._get_param('in_nprompt', '') or ''
        )
        loras = pprompt_loras + nprompt_loras
        unique = {(l['path'], l['is_high_noise']): l for l in loras}

        values = {
            'prompt': clean_pprompt,
            'negative_prompt': clean_nprompt,
            'output_format': self.extension,
            'loras': list(unique.values()) or None,
            'control_image': (self._encoded_image('in_control_img')
                              if self._get_param('in_cnnet_bool') else None),
            'ip_adapter_image': (
                self._encoded_image('in_ip_adapter_img')
                if self._get_param('in_ip_adapter_bool') else None
            ),
            **self._input_images(),
        }
        flags = {name: values[name] is not None
                 for name in ('control_image', 'ip_adapter_image')}

        # The server expects sample_params even when it's empty
        return emit.payload(
            self._context(values, flags), {'sample_params': {}}
        )

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

    def _update(self, pct=0, status="", step=None, total=None, gallery=None):
        """Builds a progress update."""
        return (
            self.fcommand,
            progress_update(pct, step, total, status),
            gr.skip(),
            gr.skip() if gallery is None else gallery,
        )

    def _final(self, stats: str, gallery=None):
        """Builds the final update, the statistics are only shown here."""
        return (
            self.fcommand,
            progress_update(visible=False),
            gr.update(visible=True, value=stats),
            gallery,
        )

    def _stopped(self):
        """Builds the update of a stopped job: keeps the command only."""
        return (
            self.fcommand,
            progress_update(visible=False),
            gr.update(visible=False, value=""),
            gr.update(value=None),
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
        remember_seed(payload.get('seed'))

        yield (
            self.fcommand,
            progress_update(0, status="Submitting..."),
            gr.update(visible=False, value=""),
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
            yield self._stopped()
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
        status = ""
        job = {}
        preview = None
        next_poll = 0.0

        while True:
            if not server_state.running:
                yield self._stopped()
                return

            # Job status and preview
            gallery = None
            if time.time() >= next_poll:
                next_poll = time.time() + self.POLL_INTERVAL
                try:
                    r = httpx.get(poll_url, timeout=5.0)
                except httpx.HTTPError:
                    r = None
                if r is not None and r.status_code in (404, 410):
                    yield self._final(
                        f"Error: job {job_id} is no longer available"
                    )
                    return
                if r is not None and r.status_code == 200:
                    job = r.json()
                    status = job.get("status", "")
                    preview = job.get("preview")

                    if status == "completed":
                        self._save_result(job.get("result") or {})
                        yield self._final(
                            self._job_stats(job, prev_stats),
                            self.outputs or None
                        )
                        return

                    if status == "cancelled":
                        yield self._stopped()
                        return

                    if status == "failed":
                        error = job.get("error") or {}
                        yield self._final(
                            f"Error: {error.get('message', 'unknown')}"
                        )
                        return

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
                shown = (0, f"Queued (position {pos})" if pos else "Queued",
                         None, None)
            elif isinstance(latest, dict) and 'percent' in latest:
                last_progress = latest
                shown = (latest['percent'], latest.get('status', ''),
                         latest.get('step'), latest.get('total'))
            elif last_progress is not None:
                shown = (last_progress['percent'],
                         last_progress.get('status', ''),
                         last_progress.get('step'),
                         last_progress.get('total'))
            elif isinstance(preview, dict) and preview.get('total_steps'):
                step = preview.get('step', 0) or 0
                total = preview['total_steps']
                shown = (int(step / total * 100), "Generating...",
                         step, total)
            else:
                shown = (0, "Starting...", None, None)
            if active_job.cancel_requested.is_set():
                shown = (shown[0], "Cancelling...", shown[2], shown[3])

            if shown != last_shown or gallery is not None:
                last_shown = shown
                yield self._update(*shown, gallery=gallery)

            time.sleep(self.PROGRESS_INTERVAL)


class Txt2ImgApiRunner(ApiTaskRunner):
    """Builds the txt2img request."""
    kind = "txt2img"
    dir_key = "txt2img_dir"
    subctrl_id = 0


class Img2ImgApiRunner(ApiTaskRunner):
    """Builds the img2img request."""
    kind = "img2img"
    dir_key = "img2img_dir"
    subctrl_id = 1

    def _input_images(self) -> Dict[str, Any]:
        values = {}
        init_img = image_to_pil(
            self._get_param('in_img_inp')
            or self._get_param('in_first_frame_inp')
        )
        if init_img is not None:
            values['init_image'] = encode_image(init_img)

        mask_input = (self._get_param('in_img_mask')
                      or self._get_param('in_mask_img'))
        mask_img = process_editor_mask(mask_input)
        if mask_img is not None:
            mask_img = mask_img.convert('L')
            if self._get_param('in_invert_mask', False):
                mask_img = ImageOps.invert(mask_img)
            values['mask_image'] = encode_image(mask_img)
        return values


class ImgEditApiRunner(ApiTaskRunner):
    """Builds the image editing (instruct) request."""
    kind = "imgedit"
    dir_key = "imgedit_dir"
    subctrl_id = 2

    def _input_images(self) -> Dict[str, Any]:
        ref_imgs = self._get_param('in_ref_img')
        if ref_imgs is None:
            ref_imgs = self._get_param('in_img_inp')
        if ref_imgs is None:
            return {}
        if not isinstance(ref_imgs, list):
            ref_imgs = [ref_imgs]

        values = {}
        encoded = []
        for item in ref_imgs:
            img = image_to_pil(item)
            if img is not None:
                encoded.append(encode_image(img))
        if encoded:
            values['ref_images'] = encoded

        mask_input = (self._get_param('in_img_mask')
                      or self._get_param('in_mask_img'))
        mask_img = process_editor_mask(mask_input)
        if mask_img is not None:
            values['mask_image'] = encode_image(mask_img.convert('L'))
        return values


class VideoApiRunner(ApiTaskRunner):
    """Builds the video request (/sdcpp/v1/vid_gen)."""
    kind = "video"
    endpoint = "/sdcpp/v1/vid_gen"
    dir_key = "any2video_dir"
    subctrl_id = 3
    extension = "webm"

    IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    def _control_frames(self) -> Optional[list]:
        """The images of the control video folder, in name order."""
        folder = self._get_param('in_control_video_dir')
        if not folder or not os.path.isdir(str(folder).strip()):
            return None
        folder = str(folder).strip()
        frames = []
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(self.IMAGE_TYPES):
                with Image.open(os.path.join(folder, name)) as img:
                    frames.append(encode_image(img.convert('RGB')))
        return frames or None

    def _audio(self) -> Optional[list]:
        """The speech audio (S2V) of the request, base64 WAV."""
        path = self._get_param('in_s2v_audio')
        if not path or not os.path.isfile(str(path)):
            return None
        with open(str(path), 'rb') as f:
            return [base64.b64encode(f.read()).decode('utf-8')]

    def _input_images(self) -> Dict[str, Any]:
        values = {}
        init = (self._get_param('in_img_inp')
                or self._get_param('in_first_frame_inp'))
        img = image_to_pil(init)
        if img is not None:
            values['init_image'] = encode_image(img.convert('RGB'))
        end = image_to_pil(self._get_param('in_last_frame_inp'))
        if end is not None:
            values['end_image'] = encode_image(end.convert('RGB'))
        values['control_frames'] = self._control_frames()
        values['ref_audios'] = self._audio()
        return {k: v for k, v in values.items() if v is not None}

    def _save_result(self, result: dict):
        """Saves the video container of a completed job."""
        b64_data = result.get("b64_json")
        if not b64_data:
            return
        fmt = result.get("output_format")
        base, _ = os.path.splitext(self.output_path)
        path = f"{base}.{fmt}" if fmt else self.output_path
        with open(path, 'wb') as f:
            f.write(decode_bytes(b64_data))
        self.output_path = path
        self.outputs.append(path)
        if result.get("frame_count") and result.get("fps"):
            self.notes.append(
                f"{result['frame_count']} frames at {result['fps']} fps")


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


def any2video_api(params: dict):
    """Creates and runs a VideoApiRunner."""
    runner = VideoApiRunner(params)
    runner.prepare()
    yield from runner.run()

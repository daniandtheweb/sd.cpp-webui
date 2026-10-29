"""Server runners: sdcpp API payloads and job lifecycle.

Payloads follow sd.cpp's SDGenerationParams::from_json_str (integer
fields must be JSON integers). Jobs run against a fake sd-server.
"""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import gradio as gr
import pytest
from PIL import Image


def _png_b64(color=(255, 0, 0), size=(8, 8), parameters=None):
    from PIL import PngImagePlugin
    buf = io.BytesIO()
    pnginfo = None
    if parameters is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", parameters)
    Image.new("RGB", size, color).save(buf, format="PNG", pnginfo=pnginfo)
    return base64.b64encode(buf.getvalue()).decode()


def _sdcpp_record(seed, prompt="a cat"):
    return {
        "schema": "sdcpp.generation/v1", "kind": "image",
        "prompt": prompt, "negative_prompt": "blurry",
        "width": 8, "height": 8, "seed": str(seed),
        "sample_params": {
            "sample_steps": 20, "sample_method": "euler_a",
            "scheduler": "karras", "guidance": {"txt_cfg": 7.0},
        },
    }


def _sdcpp_parameters(record):
    """Metadata text as sd.cpp embeds it: infotext, SDCPP: <JSON>."""
    infotext = (f"{record['prompt']}\nNegative prompt: blurry\n"
                f"Steps: 20, Sampler: Euler a, Schedule type: Karras, "
                f"CFG scale: 7, Seed: {record['seed']}, Size: 8x8")
    return f"{infotext}, SDCPP: {json.dumps(record, separators=(',', ':'))}"


@pytest.fixture(autouse=True)
def exe_name_mock(mocker):
    return mocker.patch(
        "modules.utils.sd_interface.exe_name", return_value="sd-cli"
    )


@pytest.fixture
def image_file(app_root):
    path = app_root / "input.png"
    Image.new("RGB", (8, 8), (0, 128, 255)).save(path)
    return str(path)


# ----------------------------------------------------------------------
# Payload mapping
# ----------------------------------------------------------------------

def _runner(cls_name, params):
    from modules.core.server import sdcpp_server
    base = {'in_ip': '127.0.0.1', 'in_port': 1234}
    base.update(params)
    return getattr(sdcpp_server, cls_name)(base)


def test_core_fields_are_typed_like_the_server_parser():
    payload = _runner('Txt2ImgApiRunner', {
        'in_pprompt': 'a cat', 'in_nprompt': 'blurry',
        'in_width': 768.0, 'in_height': 512.0, 'in_steps': 25.0,
        'in_batch_count': 2.0, 'in_seed': 42.0, 'in_clip_skip': 2.0,
        'in_cfg': 6.5, 'in_sampling': 'euler_a', 'in_scheduler': 'karras',
    })._build_payload()

    assert payload['prompt'] == 'a cat'
    assert payload['negative_prompt'] == 'blurry'
    for key in ('width', 'height', 'batch_count', 'seed', 'clip_skip'):
        assert type(payload[key]) is int, key
    sp = payload['sample_params']
    assert type(sp['sample_steps']) is int
    assert sp['sample_method'] == 'euler_a'
    assert sp['scheduler'] == 'karras'
    assert sp['guidance']['txt_cfg'] == 6.5
    assert payload['output_format'] == 'png'
    # No preview section in the tab -> cheap projection preview
    assert payload['preview'] == 'proj'


def test_disabled_toggles_are_omitted():
    payload = _runner('Txt2ImgApiRunner', {
        'in_guidance_bool': False, 'in_guidance': 3.5,
        'in_flow_shift_bool': False, 'in_flow_shift': 3.0,
        'in_slg_bool': False, 'in_cnnet_bool': False,
        'in_img_cfg_bool': False, 'in_img_cfg': 2.0,
    })._build_payload()
    guidance = payload['sample_params'].get('guidance', {})
    assert 'distilled_guidance' not in guidance
    assert 'img_cfg' not in guidance
    assert 'slg' not in guidance
    assert 'flow_shift' not in payload['sample_params']
    assert 'control_image' not in payload


def test_advanced_sampling_fields():
    sp = _runner('Txt2ImgApiRunner', {
        'in_guidance_bool': True, 'in_guidance': 3.5,
        'in_flow_shift_bool': True, 'in_flow_shift': 3.0,
        'in_eta_bool': True, 'in_eta': 0.5,
        'in_timestep_shift_bool': True, 'in_timestep_shift': 250.0,
        'in_sigmas': '14.6, 7.0, 3.1, 0',
        'in_slg_bool': True, 'in_slg_scale': 2.5,
        'in_skip_layer_start': 0.01, 'in_skip_layer_end': 0.2,
        'in_skip_layers': '[7,8,9]',
    })._build_payload()['sample_params']

    assert sp['guidance']['distilled_guidance'] == 3.5
    assert sp['flow_shift'] == 3.0
    assert sp['eta'] == 0.5
    assert type(sp['shifted_timestep']) is int
    assert sp['custom_sigmas'] == [14.6, 7.0, 3.1, 0.0]
    assert sp['guidance']['slg'] == {
        'scale': 2.5, 'layer_start': 0.01, 'layer_end': 0.2,
        'layers': [7, 8, 9],
    }


def test_hires_vae_tiling_cache_preview_and_preprocess():
    payload = _runner('Txt2ImgApiRunner', {
        'in_hires_bool': True, 'in_hires_upscaler': 'RealESRGAN_x4.pth',
        'in_hires_scale': 2.0, 'in_hires_width': 0.0, 'in_hires_height': 0.0,
        'in_hires_steps': 10.0, 'in_hires_denoising_strength': 0.6,
        'in_hires_upscale_tile_size': 128.0, 'in_hires_sigmas': '',
        'in_vae_tiling': True, 'in_vae_tile_overlap': 0.5,
        'in_vae_tile_size': 512.0, 'in_vae_relative_bool': False,
        'in_cache_bool': True, 'in_cache_mode': 'easycache',
        'in_cache_option': '"threshold=0.2"', 'in_scm_mask': '',
        'in_scm_policy': 'static',
        'in_preview_bool': True, 'in_preview_mode': 'tae',
        'in_preview_interval': 2.0,
        'in_img_preprocess': 'target=ref,mode=none',
    })._build_payload()

    hires = payload['hires']
    assert hires['enabled'] is True
    assert hires['upscaler'] == 'RealESRGAN_x4'  # resolved by stem
    for key in ('target_width', 'target_height', 'steps',
                'upscale_tile_size'):
        assert type(hires[key]) is int, key
    assert 'custom_sigmas' not in hires

    tiling = payload['vae_tiling_params']
    assert tiling == {
        'enabled': True, 'target_overlap': 0.5,
        'tile_size_w': 512, 'tile_size_h': 512,
    }
    assert payload['cache_mode'] == 'easycache'
    assert payload['cache_option'] == 'threshold=0.2'
    assert payload['scm_policy_dynamic'] is False
    assert 'scm_mask' not in payload
    assert payload['preview'] == 'tae'
    assert payload['preview_interval'] == 2
    assert payload['image_preprocess'] == 'target=ref,mode=none'


def test_output_settings_are_sent_per_request():
    payload = _runner('Txt2ImgApiRunner', {
        'in_disable_img_metadata': True, 'in_compression_quality': 80.0,
    })._build_payload()
    assert payload['embed_image_metadata'] is False
    assert payload['output_compression'] == 80


def test_runtime_toggles_can_switch_launch_defaults_off():
    payload = _runner('Txt2ImgApiRunner', {
        'in_vae_tiling': False, 'in_cache_bool': False,
        'in_preview_bool': False, 'in_hires_bool': False,
    })._build_payload()
    assert payload['vae_tiling_params'] == {'enabled': False}
    assert payload['cache_mode'] == 'disabled'
    assert payload['preview'] == 'none'
    assert payload['hires'] == {'enabled': False}


def test_loras_are_structured_and_high_noise_aware(mocker):
    from modules.core.server import sdcpp_server
    sdcpp_server.ApiTaskRunner._lora_lookup_cache.clear()
    mocker.patch.object(
        sdcpp_server, 'get_loras',
        return_value=['style/detail.safetensors', 'wan_hn.safetensors']
    )
    payload = _runner('Txt2ImgApiRunner', {
        'in_pprompt': ('a cat <lora:detail:0.8> '
                       '<lora:|high_noise|wan_hn:1.2>'),
    })._build_payload()

    assert '<lora' not in payload['prompt']
    assert payload['lora'] == [
        {'path': 'style/detail.safetensors', 'multiplier': 0.8,
         'is_high_noise': False},
        {'path': 'wan_hn.safetensors', 'multiplier': 1.2,
         'is_high_noise': True},
    ]


def test_controlnet_image_is_sent(image_file):
    payload = _runner('Txt2ImgApiRunner', {
        'in_cnnet_bool': True, 'in_control_img': image_file,
        'in_control_strength': 0.7,
    })._build_payload()
    assert payload['control_strength'] == 0.7
    assert Image.open(io.BytesIO(
        base64.b64decode(payload['control_image']))).size == (8, 8)


def test_img2img_inputs(image_file, app_root):
    mask_path = app_root / "mask.png"
    Image.new("L", (8, 8), 255).save(mask_path)
    payload = _runner('Img2ImgApiRunner', {
        'in_img_inp': image_file,
        'in_img_mask': {'background': str(mask_path), 'layers': []},
        'in_invert_mask': True, 'in_strength': 0.6,
        'in_img_cfg_bool': True, 'in_img_cfg': 1.5,
    })._build_payload()

    assert payload['strength'] == 0.6
    assert payload['sample_params']['guidance']['img_cfg'] == 1.5
    assert 'init_image' in payload
    mask = Image.open(io.BytesIO(base64.b64decode(payload['mask_image'])))
    assert mask.mode == 'L'
    assert mask.getextrema() == (0, 0)  # inverted


def test_imgedit_uses_native_reference_images(image_file):
    payload = _runner('ImgEditApiRunner', {
        'in_pprompt': 'make it blue',
        'in_ref_img': [(image_file, None), (image_file, None)],
        'in_increase_ref_index': True,
        'in_disable_auto_resize_ref_image': True,
    })._build_payload()

    assert len(payload['ref_images']) == 2
    assert payload['ref_image_args'] == (
        'ref_index_mode=increase,resize_before_vae=false'
    )
    assert 'init_image' not in payload


def test_displayed_payload_hides_base64(image_file):
    from modules.core.server.sdcpp_server import redact_payload
    payload = _runner('ImgEditApiRunner', {
        'in_ref_img': [(image_file, None)],
    })._build_payload()
    shown = json.dumps(redact_payload(payload))
    assert payload['ref_images'][0] not in shown
    assert '<base64' in shown


# ----------------------------------------------------------------------
# Job lifecycle against a fake sd-server
# ----------------------------------------------------------------------

class FakeServer:
    """Minimal /sdcpp/v1 implementation recording what it receives."""

    def __init__(self, polls_before_done=2, fail=None, hold=False,
                 on_poll=None):
        self.requests = []
        self.upscales = []
        self.cancelled = []
        self.polls = 0
        self.polls_before_done = polls_before_done
        self.fail = fail
        self.hold = hold
        self.on_poll = on_poll
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                length = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(length) or b"{}")

            def do_POST(self):
                if self.path == "/sdcpp/v1/img_gen":
                    fake.requests.append(self._body())
                    self._send(202, {"id": "job_1", "status": "queued"})
                elif self.path == "/sdcpp/v1/jobs/job_1/cancel":
                    fake.cancelled.append("job_1")
                    self._send(200, {"cancel_requested": True})
                elif self.path == "/sdcpp/v1/upscale":
                    body = self._body()
                    fake.upscales.append(body)
                    # Like sd.cpp: the record holds only the upscale ops,
                    # repeats merged as first op + postprocess list.
                    op = {"schema": "sdcpp.generation/v1", "kind": "image",
                          "upscale": {"upscale_factor": 4}}
                    info = dict(op, postprocess=[op] * (body["repeats"] - 1)) \
                        if body["repeats"] > 1 else op
                    self._send(200, {"generation_info": info, "images": [{
                        "index": 0,
                        "b64_json": _png_b64(
                            size=(32, 32),
                            parameters="SDCPP: " + json.dumps(info)),
                    }]})
                else:
                    self._send(404, {"error": "not found"})

            def do_GET(self):
                if self.path != "/sdcpp/v1/jobs/job_1":
                    self._send(404, {"error": "not found"})
                    return
                fake.polls += 1
                if fake.on_poll:
                    fake.on_poll(fake.polls)
                job = {"id": "job_1", "kind": "img_gen",
                       "started": 1000, "completed": None}
                if fake.cancelled:
                    job.update(status="cancelled")
                elif fake.hold or fake.polls <= fake.polls_before_done:
                    job.update(status="generating", preview={
                        "pass": 1, "step": fake.polls, "total_steps": 4,
                        "b64_json": _png_b64((0, fake.polls, 0)),
                    })
                elif fake.fail:
                    job.update(status="failed", error={
                        "code": "generation_failed", "message": fake.fail})
                else:
                    n = fake.requests[-1].get("batch_count", 1)
                    job.update(status="completed", completed=3500, result={
                        "output_format": "png",
                        "images": [{
                            "index": i,
                            "b64_json": _png_b64(parameters=_sdcpp_parameters(
                                _sdcpp_record(100 + i))),
                            "generation_info": _sdcpp_record(100 + i),
                        } for i in range(n)],
                    })
                self._send(200, job)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def fake_server(mocker, tmp_path):
    from modules.core.server import sdcpp_server
    from modules.shared_instance import server_state, config
    mocker.patch.object(sdcpp_server.ApiTaskRunner, 'POLL_INTERVAL', 0.01)
    mocker.patch.object(sdcpp_server.ApiTaskRunner, 'PROGRESS_INTERVAL', 0.01)
    mocker.patch.object(server_state, 'running', True)
    mocker.patch.object(server_state, 'last_generation_stats', '')
    # Runners target the address the tab gives
    mocker.patch.object(server_state, 'ip', '')
    # Keep generated files out of the shared output dirs other tests count.
    saved = config.data.get('txt2img_dir')
    config.update_settings({'txt2img_dir': str(tmp_path)})
    servers = []

    def make(**kwargs):
        server = FakeServer(**kwargs)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()
    config.update_settings({'txt2img_dir': saved})


def _run(api_name, port, params):
    from modules.core.server import sdcpp_server
    base = {'in_ip': '127.0.0.1', 'in_port': port, 'in_seed': -1}
    base.update(params)
    return list(getattr(sdcpp_server, api_name)(base))


def test_job_completes_and_keeps_server_metadata(fake_server, mocker):
    from modules.core.server import sdcpp_server
    mocker.patch.object(
        sdcpp_server.ApiTaskRunner, '_wait_for_log_stats', return_value='')
    server = fake_server()
    updates = _run('txt2img_api', server.port,
                   {'in_pprompt': 'a cat', 'in_batch_count': 2})

    assert server.requests[0]['batch_count'] == 2
    final = updates[-1]
    outputs = final[3]
    assert len(outputs) == 2
    # Statistics are only shown at the end
    assert final[2]['visible'] is True
    assert 'Job time: 2.50s' in final[2]['value']
    assert all(u[2] == gr.skip() for u in updates[1:-1])

    # The server's bytes, including its embedded metadata, are kept as-is
    for i, path in enumerate(outputs):
        params = Image.open(path).info['parameters']
        assert params == _sdcpp_parameters(_sdcpp_record(100 + i))

    # Preview frames were streamed to the gallery while generating
    assert any(isinstance(u[3], list) and u[3] and
               isinstance(u[3][0], Image.Image) for u in updates[:-1])


def test_progress_shows_speed_and_eta_like_cli(fake_server, mocker):
    """The server log progress (Speed | ETA) is shown while generating."""
    from modules.core.server import sdcpp_server
    from modules.shared_instance import server_state
    mocker.patch.object(
        sdcpp_server.ApiTaskRunner, '_wait_for_log_stats', return_value='')
    # Stale progress from a previous job must not be shown
    mocker.patch.object(server_state, 'latest_update',
                        {"percent": 90, "status": "old"})

    def log_progress(poll):
        if poll == 2:
            server_state.latest_update = {
                "percent": 50, "step": 10, "total": 20,
                "status": "Speed: 1.50 it/s | ETA: 00:03"}

    server = fake_server(polls_before_done=3, on_poll=log_progress)
    updates = _run('txt2img_api', server.port, {})
    from modules.ui.progress import progress_html
    shown = [u[1]['value'] for u in updates[1:-1]]

    assert all(">old<" not in bar for bar in shown)
    # Preview fallback
    assert shown[0] == progress_html(25, 1, 4, "Generating...")
    assert progress_html(
        50, 10, 20, "Speed: 1.50 it/s | ETA: 00:03") in shown


def test_failed_job_reports_server_error(fake_server):
    server = fake_server(fail="out of memory")
    final = _run('txt2img_api', server.port, {})[-1]
    assert final[2]['value'] == "Error: out of memory"


def test_post_upscale_goes_through_upscale_endpoint(fake_server, mocker):
    from modules.core.server import sdcpp_server
    mocker.patch.object(
        sdcpp_server.ApiTaskRunner, '_wait_for_log_stats', return_value='')
    server = fake_server()
    final = _run('txt2img_api', server.port, {
        'in_upscl_bool': True, 'in_upscl': 'RealESRGAN_x4plus.pth',
        'in_upscl_rep': 2.0, 'in_upscl_tile_size': 192.0,
    })[-1]

    assert server.upscales[0]['upscaler'] == 'RealESRGAN_x4plus'
    assert server.upscales[0]['repeats'] == 2
    assert server.upscales[0]['tile_size'] == 192
    upscaled = Image.open(final[3][0])
    assert upscaled.size == (32, 32)

    # Source infotext and record survive, with both upscale passes
    # appended to `postprocess` the way sd-cli would record them.
    from modules.utils.metadata_utils import find_sdcpp_generation
    infotext, record = find_sdcpp_generation(upscaled.info['parameters'])
    assert infotext.startswith("a cat\nNegative prompt: blurry")
    assert record['seed'] == "100"
    assert record['postprocess'] == [
        {"schema": "sdcpp.generation/v1", "kind": "image",
         "upscale": {"upscale_factor": 4}},
    ] * 2


def test_upscale_keeps_metadata_off_when_source_has_none():
    from modules.core.server.sdcpp_server import merge_upscale_metadata
    source = base64.b64decode(_png_b64())
    upscaled = base64.b64decode(_png_b64(size=(32, 32),
                                         parameters='SDCPP: {}'))
    merged = Image.open(io.BytesIO(
        merge_upscale_metadata(source, upscaled, {})))
    assert 'parameters' not in merged.info


def test_gallery_reads_native_sdcpp_record():
    from modules.utils.metadata_utils import extract_params_from_text
    record = _sdcpp_record(4242, prompt="a cat, SDCPP: not json {")
    text = "PNG: tEXt\nparameters: " + _sdcpp_parameters(record)
    params = extract_params_from_text(text)
    assert params == {
        'pprompt': "a cat, SDCPP: not json {", 'nprompt': "blurry",
        'steps': 20, 'sampler': "euler_a", 'scheduler': "karras",
        'cfg': 7.0, 'seed': 4242,
    }


def test_gallery_falls_back_to_infotext_without_record():
    from modules.utils.metadata_utils import extract_params_from_text
    params = extract_params_from_text(
        "a dog\nSteps: 12, CFG scale: 5.0, Seed: 7, Sampler: euler karras")
    assert params['steps'] == 12 and params['seed'] == 7
    assert params['sampler'] == 'euler'


def test_stop_cancels_running_job(fake_server):
    from modules.core.server import sdcpp_server
    server = fake_server(hold=True)
    results = {}

    def worker():
        results['updates'] = _run('txt2img_api', server.port, {})

    thread = threading.Thread(target=worker)
    thread.start()
    for _ in range(500):
        if server.polls >= 2:
            break
        threading.Event().wait(0.01)
    sdcpp_server.cancel_active_job()
    thread.join(timeout=5)

    assert server.cancelled == ["job_1"]
    # Stopped: the command stays, statistics and images are cleared
    command, progress, stats, images = results['updates'][-1]
    assert '"prompt"' in command
    assert progress['visible'] is False
    assert stats == {'__type__': 'update', 'visible': False, 'value': ""}
    assert images == {'__type__': 'update', 'value': None}
    # Nothing left to cancel once the job has finished
    assert sdcpp_server.active_job.cancel() is False


def test_server_launch_passes_context_settings(app_root):
    from modules.shared_instance import config
    from modules.core.server.manager import ServerRunner

    saved = config.data.get('upscl_dir')
    (app_root / 'upscalers').mkdir(exist_ok=True)
    try:
        config.update_settings({'upscl_dir': str(app_root / 'upscalers')})
        runner = ServerRunner({
            'in_ip': '127.0.0.1', 'in_port': 1234,
            'in_rng': 'cpu', 'in_sampler_rng': 'Default',
            'in_lora_apply': 'immediately',
            'in_qwen_image_zero_cond_t': True,
        })
        runner.build_command()
    finally:
        config.update_settings({'upscl_dir': saved})

    cmd = runner.command
    value = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
    assert value['--hires-upscalers-dir'].endswith('upscalers')
    assert value['--rng'] == 'cpu'
    assert '--sampler-rng' not in cmd
    assert value['--lora-apply-mode'] == 'immediately'
    assert value['--model-args'] == 'qwen_image_zero_cond_t=true'


def test_runner_targets_the_running_server(mocker):
    """A tab generates on the server started from another tab."""
    from modules.core.server import sdcpp_server
    from modules.shared_instance import server_state
    mocker.patch.object(server_state, 'running', True)
    mocker.patch.object(server_state, 'ip', '127.0.0.1')
    mocker.patch.object(server_state, 'port', 1234)
    runner = sdcpp_server.Img2ImgApiRunner(
        {'in_ip': '127.0.0.1', 'in_port': 9999.0})
    assert runner.base_url == "http://127.0.0.1:1234"

    mocker.patch.object(server_state, 'running', False)
    runner = sdcpp_server.Img2ImgApiRunner(
        {'in_ip': '127.0.0.1', 'in_port': 9999.0})
    assert runner.base_url == "http://127.0.0.1:9999"


def test_stop_reaches_every_tab(mocker):
    """Stopping the watcher broadcasts the final status to all tabs."""
    from modules.core.server import live_status
    from modules.shared_instance import server_state
    mocker.patch.object(server_state, 'running', False)
    mocker.patch.object(server_state, 'last_exit_code', 0, create=True)

    tabs = [live_status._subscribe() for _ in range(3)]
    try:
        live_status.stop_watcher()
        for q in tabs:
            status, btn = q.get_nowait()
            assert status == "Stopped (No Model Loaded)"
            assert btn['interactive'] is False
    finally:
        for q in tabs:
            live_status._unsubscribe(q)


def test_progress_follows_the_log_between_job_polls(fake_server, mocker):
    """The bar is not held back by the (slower) job polling."""
    from modules.core.server import sdcpp_server
    from modules.shared_instance import server_state
    mocker.patch.object(sdcpp_server.ApiTaskRunner, 'POLL_INTERVAL', 0.5)
    mocker.patch.object(server_state, 'latest_update', {})
    server = fake_server(hold=True)

    gen = sdcpp_server.txt2img_api(
        {'in_ip': '127.0.0.1', 'in_port': server.port, 'in_seed': -1})
    polls_seen = None
    for update in gen:
        if server.polls == 1 and not server_state.latest_update:
            server_state.latest_update = {
                "percent": 35, "step": 7, "total": 20,
                "status": "Speed: 2.00 it/s | ETA: 00:06"}
        if ">Step 7/20 · 35%<" in update[1]['value']:
            polls_seen = server.polls
            break
    gen.close()

    assert polls_seen == 1

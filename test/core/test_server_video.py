"""Server video tab: the vid_gen request, its inputs, the result file and
the startup command of the video models."""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

CONTAINER = b"RIFF....WEBM-bytes"


def _runner(params):
    from modules.core.server.sdcpp_server import VideoApiRunner
    base = {'in_ip': '127.0.0.1', 'in_port': 1234}
    base.update(params)
    return VideoApiRunner(base)


@pytest.fixture
def frames_dir(app_root):
    folder = app_root / 'frames'
    folder.mkdir()
    for name, color in (('b.png', (0, 255, 0)), ('a.png', (255, 0, 0))):
        Image.new('RGB', (4, 4), color).save(folder / name)
    (folder / 'notes.txt').write_text('not a frame')
    return folder


def test_video_request_fields():
    payload = _runner({
        'in_pprompt': 'a cat', 'in_width': 832.0, 'in_height': 480.0,
        'in_steps': 20.0, 'in_cfg': 6.0, 'in_seed': 5.0,
        'in_sampling': 'euler', 'in_frames': 33.0, 'in_fps': 16.0,
    })._build_payload()
    assert payload['prompt'] == 'a cat'
    assert payload['output_format'] == 'webm'
    for key in ('width', 'height', 'seed', 'video_frames', 'fps'):
        assert type(payload[key]) is int, key
    assert payload['video_frames'] == 33
    assert payload['sample_params']['sample_steps'] == 20
    # The image request fields don't belong to it
    for key in ('batch_count', 'mask_image', 'control_image'):
        assert key not in payload


def test_moe_and_vace_need_their_checkbox():
    off = _runner({
        'in_moe_boundary_bool': False, 'in_moe_boundary': 0.875,
        'in_vace_strength_bool': False, 'in_vace_strength': 1.0,
    })._build_payload()
    assert 'moe_boundary' not in off and 'vace_strength' not in off
    on = _runner({
        'in_moe_boundary_bool': True, 'in_moe_boundary': 0.9,
        'in_vace_strength_bool': True, 'in_vace_strength': 0.5,
    })._build_payload()
    assert on['moe_boundary'] == 0.9
    assert on['vace_strength'] == 0.5


def test_high_noise_settings_are_nested_and_follow_the_mode():
    params = {
        'in_high_noise_bool': True, 'in_high_noise_steps': 8.0,
        'in_high_noise_cfg': 3.0, 'in_high_noise_sampling': 'euler',
    }
    hn = _runner(dict(params, in_video_mode=0))._build_payload()[
        'high_noise_sample_params']
    assert hn['sample_steps'] == 8
    assert hn['guidance']['txt_cfg'] == 3.0
    assert hn['sample_method'] == 'euler'
    # AnimateDiff has no high noise model
    ad = _runner(dict(params, in_video_mode=1))._build_payload()
    assert 'high_noise_sample_params' not in ad
    off = _runner(dict(params, in_high_noise_bool=False))._build_payload()
    assert 'high_noise_sample_params' not in off


def test_vae_tiling_marker_is_sent_for_video():
    payload = _runner({'in_vae_tiling': True})._build_payload()
    assert payload['vae_tiling_params']['enabled'] is True
    payload = _runner({'in_vae_tiling': False})._build_payload()
    assert payload['vae_tiling_params']['enabled'] is False


def test_inputs_are_base64(image_file_pair, frames_dir, app_root):
    first, last = image_file_pair
    audio = app_root / 'voice.wav'
    audio.write_bytes(b'RIFFwav')
    values = _runner({
        'in_first_frame_inp': first, 'in_last_frame_inp': last,
        'in_control_video_dir': str(frames_dir),
        'in_s2v_audio': str(audio),
    })._input_images()
    assert set(values) == {
        'init_image', 'end_image', 'control_frames', 'ref_audios'}
    assert base64.b64decode(values['ref_audios'][0]) == b'RIFFwav'
    # sorted by name, only images
    assert len(values['control_frames']) == 2


def test_image_to_video_prefers_its_own_input(image_file_pair):
    first, last = image_file_pair
    values = _runner({
        'in_img_inp': first, 'in_first_frame_inp': last,
    })._input_images()
    assert 'init_image' in values and 'end_image' not in values


def test_nothing_is_sent_without_inputs():
    assert _runner({})._input_images() == {}
    payload = _runner({})._build_payload()
    for key in ('init_image', 'end_image', 'control_frames', 'ref_audios'):
        assert key not in payload


def test_displayed_payload_hides_audio_and_frames():
    from modules.core.server.sdcpp_server import redact_payload
    shown = redact_payload({'ref_audios': ['A' * 500],
                            'control_frames': ['B' * 500]})
    assert 'A' * 100 not in json.dumps(shown)
    assert 'B' * 100 not in json.dumps(shown)


@pytest.fixture
def image_file_pair(app_root):
    paths = []
    for name, color in (('first.png', (1, 2, 3)), ('last.png', (4, 5, 6))):
        path = app_root / name
        Image.new('RGB', (8, 8), color).save(path)
        paths.append(str(path))
    return paths


# ----------------------------------------------------------------------
# Job against a fake sd-server
# ----------------------------------------------------------------------

class FakeVideoServer:
    def __init__(self):
        self.requests = []
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

            def do_POST(self):
                if self.path != "/sdcpp/v1/vid_gen":
                    self._send(404, {})
                    return
                length = int(self.headers.get("Content-Length", 0))
                fake.requests.append(json.loads(self.rfile.read(length)))
                self._send(202, {"id": "job_1", "status": "queued"})

            def do_GET(self):
                if self.path != "/sdcpp/v1/jobs/job_1":
                    self._send(404, {})
                    return
                self._send(200, {
                    "id": "job_1", "status": "completed", "started": 1,
                    "completed": 2, "result": {
                        "b64_json": base64.b64encode(CONTAINER).decode(),
                        "mime_type": "video/webm", "output_format": "webm",
                        "fps": 16, "frame_count": 33},
                })

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()


def test_job_saves_the_video_container(mocker, tmp_path):
    from modules.core.server import sdcpp_server
    from modules.shared_instance import server_state, config
    mocker.patch.object(sdcpp_server.ApiTaskRunner, 'POLL_INTERVAL', 0.01)
    mocker.patch.object(sdcpp_server.ApiTaskRunner, 'PROGRESS_INTERVAL', 0.01)
    mocker.patch.object(server_state, 'running', True)
    mocker.patch.object(server_state, 'ip', '')
    saved = config.data.get('any2video_dir')
    config.update_settings({'any2video_dir': str(tmp_path)})
    server = FakeVideoServer()
    try:
        updates = list(sdcpp_server.any2video_api({
            'in_ip': '127.0.0.1', 'in_port': server.port, 'in_seed': -1,
            'in_frames': 33.0, 'in_fps': 16.0,
        }))
    finally:
        server.httpd.shutdown()
        config.update_settings({'any2video_dir': saved})

    assert server.requests[0]['video_frames'] == 33
    files = [p for p in tmp_path.iterdir() if p.suffix == '.webm']
    assert len(files) == 1
    assert files[0].read_bytes() == CONTAINER
    assert updates, 'the run yields updates'


# ----------------------------------------------------------------------
# Startup command
# ----------------------------------------------------------------------

def _startup(params):
    from modules.core.server.manager import ServerRunner
    runner = ServerRunner({'in_ip': '127.0.0.1', 'in_port': 1234, **params})
    runner.build_command()
    return runner


def test_video_tab_starts_the_server_with_the_video_models():
    runner = _startup({
        'in_video_mode': 0, 'f_unet_model': 'wan.gguf',
        'f_high_noise_model': 'wan_high.gguf', 'f_umt5_xxl': 't5.gguf',
        'f_unet_vae': 'vae.safetensors',
    })
    assert runner.kind == 'video'
    cmd = runner.command
    value = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
    assert value['--diffusion-model'].endswith('wan.gguf')
    assert value['--high-noise-diffusion-model'].endswith('wan_high.gguf')
    assert value['--t5xxl'].endswith('t5.gguf')
    assert value['--vae'].endswith('vae.safetensors')


def test_video_tab_starts_animatediff_models():
    runner = _startup({
        'in_video_mode': 1, 'f_animatediff_ckpt': 'ad.safetensors',
        'f_motion_module': 'motion.safetensors',
    })
    cmd = runner.command
    value = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)}
    assert value['--model'].endswith('ad.safetensors')
    assert value['--motion-module'].endswith('motion.safetensors')
    assert '--diffusion-model' not in cmd


def test_image_tabs_still_start_as_server():
    assert _startup({}).kind == 'server'

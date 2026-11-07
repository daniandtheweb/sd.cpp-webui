import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from PIL import Image

import json

from extensions.pose import detector, editor, extension, skeleton
from modules.extensions import deps
from modules.extensions.api import api_for
from modules.extensions.host import extensions
from modules.extensions.loader import load_extensions

POSE = Path(__file__).parents[2] / 'extensions' / 'pose'


def landmark(x, y, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def body(visible=None):
    """33 landmarks: landmark i is at (i/100, i/50); `visible` hides rest."""
    return [
        landmark(i / 100, i / 50, 1.0 if visible is None or i in visible
                 else 0.0)
        for i in range(33)
    ]


# ---------------------------------------------------------------- skeleton

def test_mapping_to_openpose_points():
    points = skeleton.to_openpose(body())
    assert len(points) == 18
    assert points[0] == (0.0, 0.0)                       # nose = 0
    assert points[2] == (0.12, 0.24)                     # right shoulder = 12
    assert points[5] == (0.11, 0.22)                     # left shoulder = 11
    assert points[10] == (0.28, 0.56)                    # right ankle = 28
    assert points[13] == (0.27, 0.54)                    # left ankle = 27
    assert points[14] == (0.05, 0.10)                    # right eye = 5
    assert points[17] == (0.07, 0.14)                    # left ear = 7
    # neck: the middle of the shoulders
    assert points[1] == pytest.approx((0.115, 0.23))


def test_hidden_points_are_none_and_the_neck_needs_both_shoulders():
    points = skeleton.to_openpose(body(visible={0, 11}), 0.5)
    assert points[0] is not None and points[5] is not None
    assert points[2] is None and points[1] is None
    assert sum(p is not None for p in points) == 2


def test_confidence_threshold():
    marks = body()
    marks[0] = landmark(0.5, 0.5, visibility=0.6)
    assert skeleton.to_openpose(marks, 0.5)[0] == (0.5, 0.5)
    assert skeleton.to_openpose(marks, 0.7)[0] is None


def _pixel(image, point):
    return image.getpixel((round(point[0] * image.width),
                           round(point[1] * image.height)))


def test_draw_colors_points_and_limbs_on_black():
    points = [None] * 18
    points[1], points[2] = (0.2, 0.5), (0.8, 0.5)   # neck - right shoulder
    image = skeleton.draw([points], (200, 100))
    assert image.size == (200, 100) and image.mode == 'RGB'
    assert _pixel(image, points[1]) == skeleton.COLORS[1]
    assert _pixel(image, points[2]) == skeleton.COLORS[2]
    # the limb (1, 2) has color 0, dimmed
    assert _pixel(image, (0.5, 0.5)) == tuple(
        int(c * skeleton.LIMB_BRIGHTNESS) for c in skeleton.COLORS[0])
    assert _pixel(image, (0.5, 0.1)) == (0, 0, 0)


# ---------------------------------------------------------------- detector

@pytest.fixture()
def fake_mediapipe(monkeypatch, tmp_path):
    """
    A mediapipe (Tasks API) whose landmarker finds `landmarks` (None: no
    one), and a downloaded model in tmp.
    """
    state = SimpleNamespace(
        landmarks=body(), people=None, seen=None, options=None)

    class Landmarker:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def detect(self, picture):
            state.seen = picture.data
            found = state.people if state.people is not None else (
                [state.landmarks] if state.landmarks else [])
            return SimpleNamespace(pose_landmarks=found)

    def options(**given):
        state.options = given
        return SimpleNamespace(**given)

    module = SimpleNamespace(
        Image=lambda image_format, data: SimpleNamespace(
            image_format=image_format, data=data),
        ImageFormat=SimpleNamespace(SRGB='SRGB'),
        tasks=SimpleNamespace(
            BaseOptions=lambda model_asset_path: SimpleNamespace(
                model_asset_path=model_asset_path),
            vision=SimpleNamespace(
                RunningMode=SimpleNamespace(IMAGE='IMAGE'),
                PoseLandmarkerOptions=options,
                PoseLandmarker=SimpleNamespace(
                    create_from_options=lambda opts: Landmarker()),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, 'mediapipe', module)

    model = tmp_path / 'models' / 'model.task'
    model.parent.mkdir()
    model.write_bytes(b'model')
    monkeypatch.setattr(detector, 'MODEL_PATH', model)
    return state


def test_detector_feeds_an_rgb_array_and_returns_the_landmarks(
    fake_mediapipe
):
    marks = detector.find_landmarks(Image.new('RGBA', (30, 20)))
    assert len(marks) == 1 and len(marks[0]) == 33
    assert fake_mediapipe.seen.shape == (20, 30, 3)
    options = fake_mediapipe.options
    assert options['num_poses'] == 1
    assert options['base_options'].model_asset_path.endswith('model.task')


def test_detector_without_a_person(fake_mediapipe):
    fake_mediapipe.landmarks = None
    with pytest.raises(detector.NoPerson, match='No person'):
        detector.find_landmarks(Image.new('RGB', (8, 8)))


def test_detector_needs_the_model(fake_mediapipe):
    detector.MODEL_PATH.unlink()
    with pytest.raises(FileNotFoundError, match='not downloaded'):
        detector.find_landmarks(Image.new('RGB', (8, 8)))


# ---------------------------------------------------------- the model file

@pytest.fixture()
def model_path(tmp_path, monkeypatch):
    path = tmp_path / 'models' / 'model.task'
    monkeypatch.setattr(detector, 'MODEL_PATH', path)
    return path


def _fetcher(model_path, data, calls=None):
    """A fetch(url, filename) that saves data like api.download does."""
    def fetch(url, filename):
        if calls is not None:
            calls.append((url, filename))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        (model_path.parent / filename).write_bytes(data)
    return fetch


def test_download_writes_the_model_once(model_path, monkeypatch):
    calls = []
    monkeypatch.setattr(detector, 'MODEL_FILE', model_path.name)
    fetch = _fetcher(model_path, b'x' * 2_000_000, calls)
    assert not detector.model_ready()
    assert 'Downloaded' in detector.download_model(fetch)
    assert detector.model_ready() and model_path.stat().st_size == 2_000_000
    assert 'already downloaded' in detector.download_model(fetch)
    assert calls == [(detector.MODEL_URL, model_path.name)]  # fetched once


def test_a_too_small_download_is_removed(model_path, monkeypatch):
    monkeypatch.setattr(detector, 'MODEL_FILE', model_path.name)
    with pytest.raises(OSError, match='too small'):
        detector.download_model(_fetcher(model_path, b'<html>error</html>'))
    assert not detector.model_ready()
    assert list(model_path.parent.iterdir()) == []


def test_the_model_comes_from_the_declared_host():
    from urllib.parse import urlsplit
    assert urlsplit(detector.MODEL_URL).hostname in extension.HOSTS


# ----------------------------------------------------------------- the tab

@pytest.fixture()
def pose_ext(tmp_path, monkeypatch, config, fake_mediapipe):
    """The real Pose extension loaded from a copy; outputs in tmp."""
    ext_dir = tmp_path / 'exts'
    ext_dir.mkdir()
    shutil.copytree(
        POSE, ext_dir / 'pose',
        ignore=shutil.ignore_patterns('__pycache__', 'models'))
    monkeypatch.setattr('modules.settings.defaults.CURRENT_DIR', str(tmp_path))
    # mediapipe is not installed here: pretend the requirement is met
    monkeypatch.setattr(deps, 'ensure', lambda names, confirm=None: None)
    assert load_extensions(
        SimpleNamespace(data={'ext_dir': str(ext_dir)})) == ['pose']
    monkeypatch.setattr('modules.extensions.ui.ensure_loaded', lambda: None)
    folder = tmp_path / 'outputs' / 'extensions' / 'pose'
    return api_for(extensions()[0]), folder


def _photo():
    return Image.new('RGB', (120, 90), (90, 120, 150))


def test_make_pose_saves_the_black_skeleton_and_numbers_the_files(
    pose_ext
):
    api, folder = pose_ext
    shown, status = extension.make_pose(api, _photo(), 0.5, extension.BLACK)
    assert shown.size == (120, 90)
    assert 'Saved' in status and '1 person(s)' in status
    extension.make_pose(api, _photo(), 0.5, extension.BLACK)
    assert sorted(p.name for p in folder.iterdir()) == ['1.png', '2.png']
    saved = Image.open(folder / '1.png')
    assert saved.size == (120, 90)
    assert saved.getpixel((119, 89)) == (0, 0, 0)   # black background


def test_preview_over_the_photo_but_the_saved_file_stays_black(pose_ext):
    api, folder = pose_ext
    shown, _ = extension.make_pose(api, _photo(), 0.5, extension.FADED)
    assert shown.getpixel((119, 89)) != (0, 0, 0)
    assert Image.open(folder / '1.png').getpixel((119, 89)) == (0, 0, 0)


def test_make_pose_problems_are_messages_not_crashes(
    pose_ext, fake_mediapipe, monkeypatch
):
    api, folder = pose_ext
    assert extension.make_pose(api, None, 0.5, extension.BLACK) == (
        None, 'Upload a photo first.')

    fake_mediapipe.landmarks = None
    assert extension.make_pose(api, _photo(), 0.5, extension.BLACK) == (
        None, 'No person found in the image.')

    fake_mediapipe.landmarks = body(visible={0})
    image, status = extension.make_pose(api, _photo(), 0.5, extension.BLACK)
    assert image is None and 'Only 1 point(s)' in status

    def boom(image, max_people):
        raise SystemExit(1)

    fake_mediapipe.landmarks = body()
    monkeypatch.setattr(detector, 'find_landmarks', boom)
    image, status = extension.make_pose(api, _photo(), 0.5, extension.BLACK)
    assert image is None and 'Pose detection failed: SystemExit' in status
    assert not folder.exists() or list(folder.iterdir()) == []


def test_several_people_are_drawn_on_the_same_image(
    pose_ext, fake_mediapipe, config, monkeypatch
):
    api, folder = pose_ext
    left = [landmark(x / 2, y, 1.0) for x, y in
            [(i / 100, i / 50) for i in range(33)]]
    fake_mediapipe.people = [body(), left]
    shown, status = extension.make_pose(
        api, _photo(), 0.5, extension.BLACK, 2)
    assert '2 person(s)' in status
    assert fake_mediapipe.options['num_poses'] == 2
    one = skeleton.draw([skeleton.to_openpose(body())], (120, 90))
    assert shown.tobytes() != one.tobytes()


def test_a_person_with_too_few_points_is_left_out(
    pose_ext, fake_mediapipe, config, monkeypatch
):
    api, folder = pose_ext
    fake_mediapipe.people = [body(), body(visible={0})]
    _, status = extension.make_pose(
        api, _photo(), 0.5, extension.BLACK, 2)
    assert '1 person(s)' in status


def test_download_button_reports_success_and_failure(model_path, monkeypatch):
    assert 'not downloaded' in extension.model_status()
    monkeypatch.setattr(detector, 'download_model', lambda fetch: 'ok')
    model_path.parent.mkdir()
    model_path.write_bytes(b'model')
    assert extension.download_model(None) == 'Pose model: downloaded.'

    def fail(fetch):
        raise OSError('no network')

    model_path.unlink()
    monkeypatch.setattr(detector, 'download_model', fail)
    text = extension.download_model(None)
    assert 'Could not download the model: no network' in text
    assert 'not downloaded' in text


def _download_button(api):
    with gr.Blocks() as demo:
        extension.render_tab(api)
    return next(b for b in demo.blocks.values()
                if isinstance(b, gr.Button) and b.value == 'Download the model')


def test_the_tab_starts_from_the_setting_and_sends_its_value(
    pose_ext, config, monkeypatch
):
    api, folder = pose_ext
    monkeypatch.setitem(config.data, 'ext_pose_max_people', 3)
    with gr.Blocks() as demo:
        extension.render_tab(api)
    slider = next(b for b in demo.blocks.values()
                  if getattr(b, 'label', None) == 'Max people')
    assert slider.value == 3
    click = next(f for f in demo.fns.values()
                 if any(i is slider for i in f.inputs))
    # photo, confidence, max people, and the editor it adds to
    assert len(click.inputs) == 4


def test_the_gallery_action_uses_the_setting(
    pose_ext, fake_mediapipe, config, monkeypatch, tmp_path
):
    api, folder = pose_ext
    monkeypatch.setitem(config.data, 'ext_pose_max_people', 4)
    path = tmp_path / 'photo.png'
    _photo().save(path)
    extension.pose_selected(api, SimpleNamespace(path=str(path)))
    assert fake_mediapipe.options['num_poses'] == 4


# ------------------------------------------------------------- the editor

def _value(people, size=(120, 90)):
    return json.dumps({'w': size[0], 'h': size[1], 'photo': '',
                       'people': people})


def _person(x=0.5):
    return [[x, i / 20] for i in range(18)]


def test_the_editor_value_round_trips():
    points = skeleton.to_openpose(body())
    value = editor.dump([points], (120, 90), _photo())
    assert json.loads(value)['photo'].startswith('data:image/jpeg;base64,')
    people, size = editor.load(value)
    assert size == (120, 90)
    assert people == [[tuple(p) if p else None for p in points]]


@pytest.mark.parametrize('bad', [
    'not json',
    'null',
    _value([[[0.5]] * 18]),
    '{"w": "wide", "h": 5, "people": []}',
])
def test_the_editor_value_is_checked(bad):
    with pytest.raises(ValueError):
        editor.load(bad)


def test_coordinates_are_clamped_and_missing_points_are_kept():
    person = _person()
    person[0] = [-3, 9]
    person[1] = None
    (people,), _ = editor.load(_value([person]))
    assert people[0] == (0.0, 1.0)
    assert people[1] is None


def test_detect_fills_the_editor_with_the_people(
    pose_ext, fake_mediapipe
):
    fake_mediapipe.people = [body(), body(visible={0}), body()]
    value, status = extension.detect_people(_photo(), 0.5, 3)
    people, size = editor.load(value)
    assert len(people) == 2 and size == (120, 90)    # one had a single point
    assert json.loads(value)['photo']
    assert 'Added 2' in status


def test_detect_problems_are_messages(pose_ext, fake_mediapipe):
    assert extension.detect_people(None, 0.5, 1)[1] == 'Upload a photo first.'
    fake_mediapipe.landmarks = None
    value, status = extension.detect_people(_photo(), 0.5, 1)
    assert 'No person' in status
    assert editor.load(value)[0] == []            # empty, over the photo
    assert json.loads(value)['photo']
    detector.MODEL_PATH.unlink()
    assert 'Download the model' in extension.detect_people(
        _photo(), 0.5, 1)[1]


def test_save_draws_what_the_editor_holds(pose_ext):
    api, folder = pose_ext
    shown, status = extension.save_edited(api, _value([_person(0.3)]))
    assert 'Saved' in status and shown.size == (120, 90)
    saved = Image.open(folder / '1.png')
    assert saved.getpixel((119, 89)) == (0, 0, 0)
    assert saved.tobytes() != Image.new('RGB', (120, 90)).tobytes()


def test_save_refuses_nothing_and_garbage(pose_ext):
    api, folder = pose_ext
    assert extension.save_edited(api, _value([]))[0] is None
    sparse = [None] * 18
    sparse[0] = [0.5, 0.5]
    assert 'Nothing to save' in extension.save_edited(
        api, _value([sparse]))[1]
    image, status = extension.save_edited(api, 'garbage')
    assert image is None and status.startswith('Not saved')
    assert not folder.exists() or list(folder.iterdir()) == []


def test_the_models_download_goes_through_the_api_and_is_logged(
    pose_ext, monkeypatch, tmp_path
):
    from modules.extensions import net
    from modules.extensions.host import downloads_log
    api, _ = pose_ext
    monkeypatch.setattr(net, '_resolve', lambda host: ['93.184.216.34'])
    data = [b'x' * 2_000_000, b'']

    class Response:
        headers = {'Content-Length': '2000000'}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return data.pop(0)

    asked = []
    monkeypatch.setattr(
        net, '_urlopen', lambda request: asked.append(request.full_url)
        or Response())
    monkeypatch.setattr(
        'modules.shared_instance.config.config_path',
        str(tmp_path / 'cfg.json'))
    # the model goes where the api puts files of this extension
    monkeypatch.setattr(
        detector, 'MODEL_PATH', Path(api.data_dir()) / detector.MODEL_FILE)

    assert 'Pose model: downloaded' in extension.download_model(api)
    assert asked == [detector.MODEL_URL]
    (record,) = downloads_log()
    assert record['extension'] == 'pose' and record['status'] == 'ok'
    assert record['kind'] == 'data' and record['bytes'] == 2_000_000
    assert record['host'] == 'storage.googleapis.com'


# ------------------------------------------------- the canvas in the editor

import subprocess
from pathlib import Path

EDITOR_JS = Path(__file__).parents[2] / 'extensions' / 'pose' / 'editor.js'


def _js(expression: str):
    """Run the editor's pure geometry (the <pure> block) with node."""
    node = shutil.which('node')
    if node is None:
        pytest.skip('node is not installed')
    source = EDITOR_JS.read_text(encoding='utf-8')
    block = source.split('// <pure>')[1].split('// </pure>')[0]
    out = subprocess.run(
        [node, '-e', f'{block}\nconsole.log(JSON.stringify({expression}))'],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_the_canvas_side_is_a_whole_number_in_range():
    assert _js("[canvasSide('512'), canvasSide('7'), canvasSide(''),"
               " canvasSide('abc'), canvasSide('99999'), canvasSide('64.4')]"
               ) == [512, None, None, None, 8192, 64]


def test_a_bigger_canvas_keeps_the_skeleton_size_in_pixels_and_centered():
    # a point at (100, 200) px of 200x400 -> (0.5, 0.5); the canvas gets
    # 100 px wider and 200 px taller: the point stays at the center
    got = _js("resizeCanvas([[[0.5, 0.5], null]], 200, 400, 300, 600)")
    assert got == [[[0.5, 0.5], None]]
    # a point 50 px from the left stays 50 px from the left of its
    # (centered) skeleton: it moves 50 px right on a canvas 100 px wider
    got = _js("resizeCanvas([[[0.25, 0.5]]], 200, 400, 300, 400)")
    assert got[0][0][0] == pytest.approx(100 / 300)
    assert got[0][0][1] == pytest.approx(0.5)


def test_scaling_a_skeleton_keeps_its_center_and_proportions():
    # two points 100 px apart on a 400x400 canvas; double the size
    got = _js("scalePerson([[0.375, 0.5], [0.625, 0.5]], 400, 400, 2)")
    assert got[0] == pytest.approx([0.25, 0.5])
    assert got[1] == pytest.approx([0.75, 0.5])
    assert _js("scalePerson([null], 400, 400, 2)") == [None]


def _found(x=0.5, y=0.5):
    person = [None] * 18
    person[0], person[1] = (x, y), (x, y + 0.1)
    return person


def test_adding_people_keeps_the_ones_already_there():
    first = editor.add_people(editor.blank(*editor.DEFAULT_SIZE),
                              [_found(0.2, 0.2)], _photo())
    second = editor.add_people(first, [_found(0.8, 0.8)], _photo())
    people, size = editor.load(second)
    assert len(people) == 2
    assert people[0][0] == pytest.approx((0.2, 0.2))   # untouched
    assert size == editor.load(first)[1]


def test_a_new_editor_takes_the_photos_size_and_later_ones_keep_the_canvas():
    first = editor.add_people(editor.blank(*editor.DEFAULT_SIZE), [],
                              _photo())
    assert editor.load(first)[1] == (120, 90)
    sized = editor.add_people(editor.blank(300, 300), [], _photo())
    assert editor.load(sized)[1] == (300, 300)         # the user's canvas
    again = editor.add_people(first, [], Image.new('RGB', (500, 500)))
    assert editor.load(again)[1] == (120, 90)          # now it has a photo


def test_detect_adds_to_the_people_in_the_editor(pose_ext, fake_mediapipe):
    fake_mediapipe.people = [body()]
    once, _ = extension.detect_people(_photo(), 0.5, 1)
    twice, status = extension.detect_people(_photo(), 0.5, 1, once)
    assert len(editor.load(twice)[0]) == 2
    assert 'Added 1' in status

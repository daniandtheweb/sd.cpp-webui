"""Doing a generation instead of the app, and running one from an extension."""

import os

import gradio as gr
import pytest
from PIL import Image

from modules.extensions import host
from modules.extensions.api import api_for
from modules.extensions.host import GenerationStopped, extensions
from modules.extensions.loader import load_extensions
from modules.utils import ui_events


def update(files=None, stats=None):
    """What a runner yields: command, progress, stats, images."""
    return ("cmd", "progress", stats if stats is not None else "stats", files)


@pytest.fixture()
def generator(monkeypatch):
    """Makes `generate` run a fake generation; returns what it was given."""
    seen = {}

    def install(*updates, stopped=False):
        def run(params):
            seen['params'] = params
            yield from updates
        monkeypatch.setitem(host._GENERATORS, (host._mode(), 'img2img'), run)
        monkeypatch.setattr(host, '_was_stopped', lambda: stopped)
        return seen
    return install


def test_generate_gives_the_files_of_the_final_update(generator):
    seen = generator(update(['preview.png']), update(['a.png', 'b.png']))
    assert host.generate('img2img', {'k': 1}) == ['a.png', 'b.png']
    assert seen['params'] == {'k': 1}


def test_generate_says_when_it_failed_or_was_stopped(generator):
    generator(update(None, {'value': "Error: no model"}))
    with pytest.raises(RuntimeError, match="no model"):
        host.generate('img2img', {})
    generator(update(None), stopped=True)
    with pytest.raises(GenerationStopped):
        host.generate('img2img', {})
    with pytest.raises(ValueError, match="no generation"):
        host.generate('nowhere', {})


def test_the_api_gives_images_and_removes_the_files(
    make_ext, generator, tmp_path
):
    load_extensions(make_ext({'g.py': "def register(api): pass\n"}))
    path = tmp_path / '1.png'
    Image.new('RGB', (8, 8), 'red').save(path)
    generator(update([str(path)]))
    (image,) = api_for(extensions()[0]).generate({'a': 1})
    assert image.size == (8, 8) and image.getpixel((0, 0)) == (255, 0, 0)
    assert not os.path.exists(path)


def _extension(make_ext, body):
    load_extensions(make_ext({'r.py': (
        "def register(api):\n"
        "    api.add_hook('replace_generation', replace)\n" + body)}))


def _run(params=None):
    def img2img(params):
        yield ("cmd", "normal", "stats", ['normal.png'])
    return list(ui_events._hooked(img2img)(params or {}))


def test_the_app_generates_unless_an_extension_takes_over(make_ext):
    _extension(make_ext, "def replace(ctx): return None\n")
    assert [u[3] for u in _run()] == [['normal.png']]


def test_a_replace_hook_generates_instead_and_shows_its_progress(make_ext):
    _extension(make_ext, (
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield 10, 'one'\n"
        "    yield 60, 'two'\n"
        "    return ['mine.png']\n"))
    updates = _run()
    assert [u[3] for u in updates][-1] == ['mine.png']
    assert 'normal.png' not in str([u[3] for u in updates])
    assert 'two' in str(updates[2][1])           # the progress bar text


def test_a_replacement_can_show_an_image_while_it_works(make_ext):
    _extension(make_ext, (
        "from PIL import Image\n"
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield 10, 'a', Image.new('RGB', (4, 4))\n"
        "    yield 50, 'b'\n"
        "    return ['mine.png']\n"))
    shown = [u[3] for u in _run()]
    assert shown[1][0].size == (4, 4) and len(shown[1]) == 1
    assert shown[-1] == ['mine.png']


def test_a_failing_or_stopped_replacement_ends_cleanly(make_ext, capsys):
    _extension(make_ext, (
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield 0, 'x'\n"
        "    raise ValueError('boom')\n"))
    last = _run()[-1]
    assert "Error: boom" in str(last[2]) and last[3] is None
    host._reset()
    from modules.extensions import loader
    loader._reset()
    _extension(make_ext, (
        "from modules.extensions.host import GenerationStopped\n"
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield 0, 'x'\n"
        "    raise GenerationStopped()\n"))
    assert "Error" not in str(_run()[-1])


def test_the_history_of_an_extensions_run_is_not_kept(
    generator, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for name in ('commands', 'pprompts', 'nprompts'):
        (tmp_path / 'outputs' / name).mkdir(parents=True)
    kept = tmp_path / 'outputs' / 'commands' / 'command_0.txt'
    kept.write_text('the user\'s own run')

    def run(params):
        (tmp_path / 'outputs' / 'commands' / 'command_1.txt').write_text('x')
        (tmp_path / 'outputs' / 'pprompts' / 'pprompt_1.txt').write_text('x')
        yield update(['a.png'])

    monkeypatch.setitem(host._GENERATORS, (host._mode(), 'img2img'), run)
    monkeypatch.setattr(host, '_was_stopped', lambda: False)
    assert host.generate('img2img', {}) == ['a.png']
    assert [p.name for p in (tmp_path / 'outputs').rglob('*.txt')] == [
        'command_0.txt']


def test_default_params_are_what_a_tab_starts_with(
    make_ext, config, monkeypatch
):
    load_extensions(make_ext({'d.py': "def register(api): pass\n"}))
    monkeypatch.setitem(config.data, 'def_steps', 41)
    params = api_for(extensions()[0]).default_params()
    assert params['in_steps'] == 41                  # the user's own default
    assert 'in_sampling' in params and 'in_pprompt' not in params


def test_the_api_lists_models_and_choices(
    make_ext, tmp_path, monkeypatch, config
):
    load_extensions(make_ext({'m.py': "def register(api): pass\n"}))
    api = api_for(extensions()[0])
    (tmp_path / 'loras').mkdir()
    (tmp_path / 'loras' / 'a.safetensors').write_bytes(b'x')
    monkeypatch.setitem(config.data, 'lora_dir', str(tmp_path / 'loras'))
    assert api.models('lora') == ['a.safetensors']
    with pytest.raises(ValueError, match="unknown kind"):
        api.models('nothing')
    assert 'euler' in api.choices('samplers')


def test_each_mode_runs_its_own_generation(monkeypatch):
    """The app imports the tabs of both modes: they must not mix."""
    def img2img(params):
        yield update(['cli.png'])

    def img2img_api(params):
        yield update(['server.png'])

    monkeypatch.setattr(host, '_GENERATORS', {})
    monkeypatch.setattr(host, '_was_stopped', lambda: False)
    host.register_generator('img2img', img2img)
    host.register_generator('img2img', img2img_api)       # imported last
    monkeypatch.setattr(host, '_mode', lambda: 'cli')
    assert host.generate('img2img', {}) == ['cli.png']
    monkeypatch.setattr(host, '_mode', lambda: 'server')
    assert host.generate('img2img', {}) == ['server.png']
    with pytest.raises(ValueError, match="in server mode"):
        host.generate('upscale', {})


def test_generating_passes_on_what_the_app_would_show(generator):
    generator(update(['preview.png']), update(['a.png']))
    steps = host.generating('img2img', {})
    shown = []
    while True:
        try:
            shown.append(next(steps))
        except StopIteration as done:
            result = done.value
            break
    assert [u[3] for u in shown] == [['preview.png'], ['a.png']]
    assert result == ['a.png']


def test_a_replacement_passes_on_the_updates_of_the_app(make_ext):
    _extension(make_ext, (
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield ('mine', 'the app progress', 'stats', ['p.png'])\n"
        "    yield 50, 'then mine'\n"
        "    return ['done.png']\n"))
    updates = _run()
    assert ('mine', 'the app progress', 'stats', ['p.png']) in updates
    assert updates[-1][3] == ['done.png']


def test_a_replacement_adds_a_second_bar_and_keeps_the_main_one(make_ext):
    _extension(make_ext, (
        "from modules.extensions.host import Progress\n"
        "from modules.ui.progress import progress_update\n"
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    # the app's own last update hides its bar: it must stay\n"
        "    yield ('cmd', progress_update(visible=False), 'stats', ['o.png'])\n"
        "    yield Progress(40, 'Tiles 1 of 4', None, (25, 'Tile 1 · step 5/20'))\n"
        "    yield Progress(50, 'Tiles 2 of 4', 'shown.png')\n"
        "    return ['done.png']\n"))
    updates = _run()
    app, with_sub, with_image = updates[1], updates[2], updates[3]
    assert app[1] == gr.skip()                   # the bar did not disappear
    assert "gen-progress-sub" in with_sub[1]['value']
    assert "Tile 1 · step 5/20" in with_sub[1]['value']
    assert with_sub[3] == gr.skip()              # no image: keeps the one shown
    assert with_image[3] == ['shown.png']
    assert "gen-progress-sub" not in with_image[1]['value']   # and no 2nd bar


def test_stopping_a_replacement_keeps_what_is_shown(make_ext):
    _extension(make_ext, (
        "from modules.extensions.host import GenerationStopped\n"
        "def replace(ctx):\n"
        "    return work()\n"
        "def work():\n"
        "    yield 0, 'x', 'shown.png'\n"
        "    raise GenerationStopped()\n"))
    last = _run()[-1]
    assert last[1]['visible'] is False and last[3] == gr.skip()

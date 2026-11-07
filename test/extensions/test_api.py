"""The API functions of version 1: settings, data_dir, hooks, gallery."""

import importlib
import os
import sys

import gradio as gr
import pytest
from PIL import Image

from modules.extensions.api import api_for
from modules.extensions.host import (
    extensions,
    generation_tab,
    run_gallery_action,
    run_post,
    run_pre,
)
from modules.extensions.loader import load_extensions
from modules.extensions.ui import render_extensions_info


@pytest.fixture()
def api(make_ext, config, mocker):
    """The api object of a loaded extension called 'demo'."""
    mocker.patch.object(config, 'save_config')
    load_extensions(make_ext({'demo.py': "def register(api): pass\n"}))
    yield api_for(extensions()[0])
    for key in [k for k in config.data if k.startswith('ext_demo_')]:
        del config.data[key]


# --------------------------------------------------------------- settings

def test_add_setting_returns_the_value_and_describes_it(api):
    assert api.add_setting('per_row', 3, label='Per row', info='Images') == 3
    spec = extensions()[0].settings['per_row']
    assert (spec.label, spec.info, spec.default) == ('Per row', 'Images', 3)
    # a label is made from the key when none is given
    api.add_setting('max_size', 5)
    assert extensions()[0].settings['max_size'].label == 'Max size'


def test_add_setting_keeps_what_is_stored(api, config):
    config.data['ext_demo_level'] = 7
    assert api.add_setting('level', 1) == 7
    assert config.data['ext_demo_level'] == 7


@pytest.mark.parametrize('default', [
    None,
    b'x',
])
def test_a_setting_is_a_bool_int_float_or_str(api, default):
    with pytest.raises(TypeError):
        api.add_setting('bad', default)


@pytest.mark.parametrize('default, value, stored', [
    (1, 5, 5),
    ('a', 'text', 'text'),
])
def test_set_setting_stores_a_value_of_the_same_type(
    api, config, default, value, stored
):
    api.add_setting('s', default)
    result = api.set_setting('s', value)
    assert (result, type(result)) == (stored, type(default))
    assert config.data['ext_demo_s'] == stored
    assert api.get_setting('s') == stored
    config.save_config.assert_called()


@pytest.mark.parametrize('default, value', [
    (1, 5.5),
    (1.5, 'x'),
    ('a', None),
])
def test_set_setting_rejects_a_value_of_another_type(
    api, config, default, value
):
    api.add_setting('s', default)
    with pytest.raises(ValueError):
        api.set_setting('s', value)
    assert config.data['ext_demo_s'] == default


def test_set_setting_checks_the_choices(api):
    api.add_setting('mode', 'a', choices=['a', 'b'])
    assert api.set_setting('mode', 'b') == 'b'
    with pytest.raises(ValueError, match='not one of'):
        api.set_setting('mode', 'z')


def test_an_extension_can_only_set_what_it_declared(api, config):
    with pytest.raises(KeyError):
        api.set_setting('not_declared', 1)
    assert 'ext_demo_not_declared' not in config.data


def test_a_bad_stored_value_falls_back_to_the_default(api, config):
    api.add_setting('n', 4)
    config.data['ext_demo_n'] = 'edited by hand'
    assert api.get_setting('n') == 4
    # and is only replaced when the user saves a new one
    assert config.data['ext_demo_n'] == 'edited by hand'


def test_get_setting_of_something_undeclared_is_an_error(api, config):
    config.data['ext_demo_raw'] = [1, 2]
    with pytest.raises(KeyError):
        api.get_setting('raw')


# ---------------------------------------------------------------- data_dir

def test_data_dir_is_next_to_the_config(api, config):
    path = api.data_dir()
    expected = os.path.join(
        os.path.dirname(os.path.abspath(config.config_path)),
        'extensions', 'demo')
    assert path == expected and os.path.isdir(path)
    assert api.data_dir() == path


# ----------------------------------------------------------------- hooks

@pytest.mark.parametrize('function, tab', [
    ('txt2img_api', 'txt2img'),
])
def test_generation_tab_of_a_function(function, tab):
    assert generation_tab(function) == tab


def test_the_pipeline_tells_the_hooks_its_tab(make_ext):
    from modules.utils.ui_events import _hooked
    load_extensions(make_ext({'t.py': (
        "def register(api):\n"
        "    api.add_hook('pre_generation',\n"
        "        lambda ctx: ctx.params.update(tab=ctx.tab))\n"
    )}))

    def imgedit_api(params):
        yield (None, None, None, ['a.png'])

    params = {}
    list(_hooked(imgedit_api)(params))
    assert params['tab'] == 'imgedit'


def test_post_hooks_chain_on_the_images(make_ext):
    load_extensions(make_ext({
        'a.py': ("PRIORITY = 0\n"
                 "def register(api):\n"
                 "    api.add_hook('post_generation',\n"
                 "        lambda ctx: setattr(ctx, 'images', ctx.images + ['A']))\n"),
        'b.py': ("PRIORITY = 1\n"
                 "def register(api):\n"
                 "    api.add_hook('post_generation',\n"
                 "        lambda ctx: setattr(ctx, 'images', ctx.images + ['B']))\n"),
    }))
    assert run_post(['x'], {}, 'txt2img') == ['x', 'A', 'B']


# ------------------------------------------------------- gallery actions

def _png(tmp_path, name='1.png'):
    path = tmp_path / name
    Image.new('RGB', (4, 4)).save(path)
    return str(path)


def test_a_gallery_action_needs_a_function_and_a_label(api):
    with pytest.raises(TypeError):
        api.add_gallery_action('Label', 'not callable')
    for label in ('', '   '):
        with pytest.raises(ValueError):
            api.add_gallery_action(label, print)
    api.add_gallery_action('  Spaced  ', print)
    assert extensions()[0].gallery_actions[0][0] == 'Spaced'


def test_an_action_gets_the_selected_image_and_returns_a_message(
    make_ext, tmp_path
):
    load_extensions(make_ext({'a.py': (
        "def register(api):\n"
        "    api.add_gallery_action('Name',\n"
        "        lambda ctx: 'saw ' + ctx.path.split('/')[-1])\n"
    )}))
    ext = extensions()[0]
    fn = ext.gallery_actions[0][1]
    assert run_gallery_action(ext, fn, _png(tmp_path, 'pic.png')) == 'saw pic.png'


@pytest.mark.parametrize('path', [
    None,
])
def test_an_action_needs_an_existing_selection(make_ext, path):
    load_extensions(make_ext({'a.py': (
        "def register(api): api.add_gallery_action('Go', lambda ctx: 'ran')\n"
    )}))
    ext = extensions()[0]
    assert run_gallery_action(
        ext, ext.gallery_actions[0][1], path) == 'Select an image first.'


def test_an_action_never_runs_on_a_video(make_ext, tmp_path):
    load_extensions(make_ext({'a.py': (
        "def register(api): api.add_gallery_action('Go', lambda ctx: 'ran')\n"
    )}))
    ext = extensions()[0]
    video = tmp_path / '1.mp4'
    video.write_bytes(b'x')
    assert 'images only' in run_gallery_action(
        ext, ext.gallery_actions[0][1], str(video))


def test_a_failing_action_is_a_message_not_a_crash(make_ext, tmp_path, capsys):
    load_extensions(make_ext({'a.py': (
        "def boom(ctx): raise ValueError('nope')\n"
        "def register(api): api.add_gallery_action('Go', boom)\n"
    )}))
    ext = extensions()[0]
    message = run_gallery_action(
        ext, ext.gallery_actions[0][1], _png(tmp_path))
    assert message == 'Failed: ValueError: nope'
    out = capsys.readouterr().out
    assert 'gallery action boom failed' in out


def test_the_gallery_runs_actions_on_the_servers_selection(
    make_ext, tmp_path, monkeypatch
):
    """The button uses the selection the server holds, not browser input."""
    cfg = make_ext({'act.py': (
        "def register(api):\n"
        "    api.add_gallery_action('Say it',\n"
        "        lambda ctx: 'saw ' + ctx.path.split('/')[-1])\n"
    )})
    # Import the Gallery without loading the real extensions folder
    monkeypatch.setattr('modules.extensions.ui.ensure_loaded', lambda: None)
    from modules.interfaces.common import gallery_tab
    monkeypatch.setattr(
        'modules.extensions.ui.ensure_loaded', lambda: load_extensions(cfg))
    try:
        tab = importlib.reload(gallery_tab)
        button = next(b for b in tab.gallery_block.blocks.values()
                      if isinstance(b, gr.Button) and b.value == 'Say it')
        handler = next(fn for fn in tab.gallery_block.fns.values()
                       if (button._id, 'click') in fn.targets)
        assert handler.inputs == []     # nothing comes from the browser

        assert handler.fn() == 'Select an image first.'
        tab.gallery_manager.current_media_path = _png(tmp_path, 'sel.png')
        assert handler.fn() == 'saw sel.png'
    finally:
        # rebuild the Gallery as it is without extensions
        monkeypatch.setattr('modules.extensions.ui.ensure_loaded', lambda: None)
        importlib.reload(gallery_tab)


# ------------------------------------------------------------------ the UI

def _build_info(make_ext):
    cfg = make_ext({'ui.py': (
        "def register(api):\n"
        "    api.add_setting('per_row', 3, label='Per row')\n"
        "    api.add_setting('title', 'x', label='Title')\n"
        "    api.add_setting('mode', 'a', choices=['a', 'b'])\n"
        "    api.add_setting('flag', False)\n"
        "    api.add_setting('ratio', 0.5)\n"
    ), 'plain.py': "def register(api): pass\n"})
    load_extensions(cfg)
    with gr.Blocks() as demo, gr.Tabs():
        render_extensions_info()
    return demo


def test_the_extensions_tab_edits_the_declared_settings(
    make_ext, config, mocker
):
    mocker.patch.object(config, 'save_config')
    mocker.patch('modules.extensions.ui.ensure_loaded')
    demo = _build_info(make_ext)

    kinds = {b.label: type(b).__name__ for b in demo.blocks.values()
             if getattr(b, 'label', None) in
             ('Per row', 'Title', 'Mode', 'Flag', 'Ratio')}
    assert kinds == {'Per row': 'Number', 'Title': 'Textbox',
                     'Mode': 'Dropdown', 'Flag': 'Checkbox',
                     'Ratio': 'Number'}
    # only the extension that has settings gets a section
    sections = [b.label for b in demo.blocks.values()
                if isinstance(b, gr.Accordion)]
    assert sections == ['ui settings']

    button = next(b for b in demo.blocks.values()
                  if isinstance(b, gr.Button) and b.value == 'Save')
    save = next(fn for fn in demo.fns.values()
                if (button._id, 'click') in fn.targets).fn
    assert save(5.0, 'new', 'b', True, 0.25) == 'Saved.'
    assert config.data['ext_ui_per_row'] == 5
    assert isinstance(config.data['ext_ui_per_row'], int)
    assert config.data['ext_ui_mode'] == 'b'
    # a bad value saves nothing at all
    before = dict(config.data)
    message = save(9, 'newer', 'zzz', False, 0.9)
    assert message.startswith('Not saved:') and 'not one of' in message
    assert config.data == before
    for key in [k for k in config.data if k.startswith('ext_ui_')]:
        del config.data[key]


# ------------------------------------- an extension must not stop the app

def test_a_hook_that_exits_does_not_stop_the_worker(make_ext, capsys):
    """sys.exit() raises SystemExit, which `except Exception` lets through."""
    cfg = make_ext({'quit.py': (
        "import sys\n"
        "SEEN = []\n"
        "def register(api):\n"
        "    api.add_hook('pre_generation', lambda ctx: sys.exit(1))\n"
        "    api.add_hook('pre_generation', lambda ctx: SEEN.append('pre'))\n"
        "    api.add_hook('post_generation', lambda ctx: sys.exit(2))\n"
        "    api.add_hook('post_generation', lambda ctx: SEEN.append('post'))\n"
    )})
    load_extensions(cfg)
    run_pre({}, 'txt2img')
    run_post(['a.png'], {}, 'txt2img')
    assert sys.modules['sdcpp_ext_quit'].SEEN == ['pre', 'post']
    out, err = capsys.readouterr()
    assert 'pre_generation hook <lambda> failed: SystemExit' in out
    assert 'post_generation hook <lambda> failed: SystemExit' in out
    assert 'Traceback' in err                # the author gets the traceback


# ------------------------------------------- setting names can't collide

def test_a_setting_cannot_take_an_app_setting_or_a_bad_name(api):
    for bad in ("", "a b", "x-y", "ü"):
        with pytest.raises(ValueError, match='not a setting name'):
            api.add_setting(bad, 1)


def test_a_setting_cannot_be_another_extensions_config_key(make_ext):
    """ext_a + b_c and ext_a_b + c are the same key in the config."""
    cfg = make_ext({
        'a.py': "def register(api): api.add_setting('b_c', 1)\n",
        'a_b.py': "def register(api): api.add_setting('c', 1)\n",
    })
    assert load_extensions(cfg) == ['a']
    other = next(e for e in extensions() if e.id == 'a_b')
    assert other.status == 'failed'
    assert "'ext_a_b_c' is already a setting of 'a'" in other.reason


def test_a_setting_cannot_be_an_app_config_key(make_ext):
    cfg = make_ext({'dir.py': "def register(api): api.add_setting('', 1)\n"})
    load_extensions(cfg)
    assert extensions()[0].status == 'failed'


# ------------------------------------------------ outputs that aren't images

def test_a_text_output_has_a_folder_and_no_gallery_button(
    make_ext, monkeypatch, tmp_path
):
    monkeypatch.setattr('modules.settings.defaults.CURRENT_DIR', str(tmp_path))
    load_extensions(make_ext({'refine.py': (
        "def register(api):\n"
        "    api.add_output('prompts', gallery=False)\n"
        "    api.add_output('cards', label='Cards')\n"
    )}))
    (ext,) = extensions()
    assert ext.status == 'loaded'
    assert list(ext.outputs.values()) == ['Cards']      # only the images
    assert os.path.isdir(ext.output_dir + 'prompts')    # but both exist


def test_text_files_are_numbered_so_none_is_overwritten(
    make_ext, monkeypatch, tmp_path
):
    monkeypatch.setattr('modules.settings.defaults.CURRENT_DIR', str(tmp_path))
    load_extensions(make_ext({'refine.py': (
        "def register(api):\n"
        "    global API; API = api\n"
        "    api.add_output(gallery=False)\n"
    )}))
    import sys
    api = sys.modules['sdcpp_ext_refine'].API
    seen = []
    for _ in range(3):
        path = api.next_output_path(extension='.txt')
        assert not os.path.exists(path)
        with open(path, 'w') as f:
            f.write('a refined prompt')
        seen.append(os.path.basename(path))
    assert seen == ['1.txt', '2.txt', '3.txt']

import os

import pytest

from modules.extensions.api import api_for
from modules.extensions.host import (
    extensions,
    registered_outputs,
    registered_tabs,
    registered_uis,
    run_post,
    run_pre,
)
from modules.extensions import loader
from modules.extensions.loader import load_extensions

GOOD = '''
NAME = 'Good'
VERSION = '1.2'
AUTHOR = 'Me'
URL = 'https://example.com'
DESCRIPTION = 'Does good.'

def replace_images(ctx):
    ctx.images = ['replaced.png']

def register(api):
    api.add_tab('Good Tab', lambda: None)
    api.add_hook('pre_generation', lambda ctx: ctx.params.update({'pre': True}))
    api.add_hook('post_generation', replace_images)
'''


def statuses():
    return {ext.id: ext.status for ext in extensions()}


def test_single_file_and_metadata(make_ext):
    cfg = make_ext({'good.py': GOOD})
    assert load_extensions(cfg) == ['good']
    (ext,) = extensions()
    assert (ext.name, ext.meta.version, ext.meta.author) == (
        'Good', '1.2', 'Me')
    assert ext.meta.url == 'https://example.com'
    assert ext.meta.description == 'Does good.'
    assert [t for t, _ in registered_tabs()] == ['Good Tab']


def test_metadata_read_without_importing(make_ext):
    cfg = make_ext({'boom.py': "NAME = 'Boom'\nVERSION = '3'\nraise OSError\n"})
    load_extensions(cfg)
    (ext,) = extensions()
    assert ext.status == 'failed'
    assert ext.name == 'Boom' and ext.meta.version == '3'
    assert 'import: OSError' in ext.reason


def test_folder_extension_with_relative_import(make_ext):
    cfg = make_ext({
        'pack/extension.py': (
            "from . import helper\n"
            "NAME = 'Pack'\n"
            "def register(api):\n"
            "    api.add_tab(helper.TITLE, lambda: None)\n"
        ),
        'pack/helper.py': "TITLE = 'From Helper'\n",
        'empty_dir/readme.txt': 'no entry file',
    })
    assert load_extensions(cfg) == ['pack']
    assert [t for t, _ in registered_tabs()] == ['From Helper']


def test_failures_are_isolated(make_ext, capsys):
    cfg = make_ext({
        'a_good.py': GOOD,
        'b_import.py': 'raise ImportError("nope")\n',
        'c_register.py': 'def register(api):\n    raise ValueError("x")\n',
        'd_exit.py': 'import sys\nsys.exit(1)\n',
        'e_noreg.py': 'X = 1\n',
        'f_newer.py': 'API_VERSION = 99\ndef register(api): pass\n',
        'g_hook.py': (
            "def register(api):\n    api.add_hook('nope', print)\n"
        ),
    })
    assert load_extensions(cfg) == ['a_good']
    assert statuses() == {
        'a_good': 'loaded', 'b_import': 'failed', 'c_register': 'failed',
        'd_exit': 'failed', 'e_noreg': 'failed', 'f_newer': 'skipped',
        'g_hook': 'failed',
    }
    reasons = {e.id: e.reason for e in extensions()}
    assert 'SystemExit' in reasons['d_exit']
    assert 'targets API v99' in reasons['f_newer']
    assert 'unknown hook' in reasons['g_hook']
    out = capsys.readouterr().out
    assert 'Failed to load 5 extension' in out
    assert '  - b_import: import: ImportError: nope' in out


def test_failed_register_leaves_nothing_behind(make_ext):
    cfg = make_ext({'half.py': (
        "def register(api):\n"
        "    api.add_tab('Half', lambda: None)\n"
        "    api.add_hook('pre_generation', lambda c: c.params.update(x=1))\n"
        "    raise RuntimeError('late')\n"
    )})
    load_extensions(cfg)
    params = {}
    run_pre(params)
    assert params == {} and registered_tabs() == []


def test_hooks(make_ext):
    cfg = make_ext({
        'bad.py': (
            "def boom(*a): raise ValueError('x')\n"
            "def register(api):\n"
            "    api.add_hook('pre_generation', boom)\n"
            "    api.add_hook('post_generation', boom)\n"
        ),
        'good.py': GOOD,
    })
    load_extensions(cfg)
    params = {}
    run_pre(params)  # the failing hook is isolated
    assert params == {'pre': True}
    assert run_post(['a.png'], {}) == ['replaced.png']


def test_ui_registration_and_params(make_ext):
    cfg = make_ext({'my-ext.py': (
        "def render(tab): return {}\n"
        "def register(api):\n"
        "    api.add_ui(render, tabs=('txt2img',))\n"
        "    api.add_hook('pre_generation', lambda ctx: ctx.params.update(\n"
        "        seen=api.param(ctx.params, 'strength', 7)))\n"
    )})
    load_extensions(cfg)
    assert [e.id for e, _ in registered_uis('txt2img')] == ['my_ext']
    assert registered_uis('img2img') == []
    params = {'ext_my_ext_strength': 3}
    run_pre(params)
    assert params['seen'] == 3
    params = {}
    run_pre(params)
    assert params['seen'] == 7


def test_loads_once(make_ext):
    cfg = make_ext({'good.py': GOOD})
    assert load_extensions(cfg) == ['good']
    assert load_extensions(cfg) == ['good']
    assert len(extensions()) == 1


def test_missing_dir_falls_back(make_ext, tmp_path, monkeypatch, capsys):
    from modules.extensions import loader
    monkeypatch.setattr(loader, '_DEFAULT_EXT_DIR', str(tmp_path / 'default'))
    cfg = make_ext({})
    cfg.data['ext_dir'] = str(tmp_path / 'missing')
    assert load_extensions(cfg) == []
    out = capsys.readouterr().out
    assert 'missing; using' in out
    assert (tmp_path / 'default').is_dir()


def test_disabled_without_flag(make_ext, monkeypatch, tmp_path):
    from modules.extensions import deps, loader
    monkeypatch.setattr(loader, 'ENABLED', False)
    monkeypatch.setattr(deps, 'install', lambda r: (_ for _ in ()).throw(
        AssertionError('installed')))
    monkeypatch.setattr(loader, '_DEFAULT_EXT_DIR', str(tmp_path / 'never'))
    cfg = make_ext({'needs.py': (
        "REQUIRES = ['sdcpp_fake_pkg>=1']\nraise SystemExit\n")})
    assert load_extensions(cfg) == []
    assert extensions() == []
    cfg.data['ext_dir'] = ''
    loader._reset()
    assert load_extensions(cfg) == []
    assert not (tmp_path / 'never').exists()


@pytest.fixture()
def outputs_root(tmp_path, monkeypatch):
    """Point the app root at a temp folder, so outputs/ lands there."""
    monkeypatch.setattr('modules.settings.defaults.CURRENT_DIR', str(tmp_path))
    return tmp_path / 'outputs' / 'extensions'


def test_add_output_declares_a_folder(make_ext, outputs_root):
    cfg = make_ext({'painter.py': (
        "def register(api):\n"
        "    api.add_output(label='Paintings')\n"
        "    api.add_output('Thumbs')\n"
    ), 'plain.py': "def register(api): pass\n"})
    load_extensions(cfg)
    outs = registered_outputs()
    assert [(e.id, label) for e, label, _ in outs] == [
        ('painter', 'Paintings'), ('painter', 'Thumbs')]
    folders = [folder for _, _, folder in outs]
    assert folders == [str(outputs_root / 'painter') + '/',
                       str(outputs_root / 'painter' / 'thumbs') + '/']
    assert all(os.path.isdir(folder) for folder in folders)


@pytest.mark.parametrize('name, folder', [
    ('../../evil', 'evil'),
    ('Nice Name!', 'nice_name'),
])
def test_output_names_cannot_escape(
    make_ext, outputs_root, tmp_path, name, folder
):
    cfg = make_ext({'x.py': (
        f"def register(api): api.add_output({name!r})\n")})
    load_extensions(cfg)
    ((_, _, path),) = registered_outputs()
    assert path == str(outputs_root / 'x' / folder) + '/'
    assert os.path.commonpath([path, str(outputs_root)]) == str(outputs_root)


def test_next_output_path_numbers_like_the_app(make_ext, outputs_root):
    cfg = make_ext({'num.py': "def register(api): api.add_output()\n"})
    load_extensions(cfg)
    api = api_for(extensions()[0])
    assert api.next_output_path() == str(outputs_root / 'num' / '1.png')
    (outputs_root / 'num' / '7_a.png').touch()
    (outputs_root / 'num' / 'cat.png').touch()
    assert api.next_output_path(extension='.webp').endswith('/8.webp')
    # a name can't leave the extension's folder here either
    assert api.next_output_path('../../x').startswith(
        str(outputs_root / 'num' / 'x'))


@pytest.mark.parametrize('extension', [
    '/../../x.png',
    '',
    None,
])
def test_a_file_extension_cannot_escape_the_folder(
    make_ext, outputs_root, extension
):
    cfg = make_ext({'esc.py': "def register(api): api.add_output()\n"})
    load_extensions(cfg)
    api = api_for(extensions()[0])
    with pytest.raises(ValueError, match='extension must look like'):
        api.next_output_path(extension=extension)


@pytest.mark.parametrize('argv, given', [
    (['app.py', '--extensions'], True),
    (['app.py', '--server', '--extens'], True),
    (['app.py', '--'], False),
    (['--extensions'], False),
])
def test_the_flag_is_read_the_way_argparse_reads_it(argv, given):
    assert loader._flag_given(argv) is given


def test_a_loose_file_named_like_a_folders_entry_is_not_an_extension(
    make_ext
):
    cfg = make_ext({'extension.py': "def register(api): pass\n",
                    'real.py': "def register(api): pass\n"})
    assert load_extensions(cfg) == ['real']


def test_hosts_and_requires_may_carry_reasons(make_ext):
    cfg = make_ext({'r.py': (
        "HOSTS = {'A.example.com': ' the search '}\n"
        "REQUIRES = {'pillow>=10': 'reads images'}\n"
        "def register(api): pass\n")})
    load_extensions(cfg)
    (ext,) = extensions()
    assert ext.meta.hosts == ['a.example.com']
    assert ext.meta.host_reasons == {'a.example.com': 'the search'}
    assert ext.meta.requires == ['pillow>=10']
    assert ext.meta.require_reasons == {'pillow>=10': 'reads images'}


def test_a_reason_that_is_not_text_is_a_problem(make_ext, capsys):
    cfg = make_ext({'r.py': (
        "HOSTS = {'a.example.com': 3}\ndef register(api): pass\n")})
    load_extensions(cfg)
    assert extensions()[0].meta.hosts == []
    assert 'HOSTS must be a string, a list of strings or a dict' in (
        capsys.readouterr().out)

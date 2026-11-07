
import gradio as gr

from modules.extensions.loader import load_extensions
from modules.extensions.ui import (
    create_extensions_ui,
    render_extension_tabs,
    render_extensions_info,
)


def test_tabs_dedup(make_ext, monkeypatch, capsys):
    cfg = make_ext({'t.py': (
        "def register(api):\n"
        "    api.add_tab('Dup', lambda: None)\n"
        "    api.add_tab('Dup', lambda: None)\n"
        "    api.add_tab('Solo', lambda: None)\n"
    )})
    monkeypatch.setattr(
        'modules.extensions.ui.ensure_loaded', lambda: load_extensions(cfg))
    with gr.Blocks(), gr.Tabs():
        render_extension_tabs()
    out = capsys.readouterr().out
    assert "Skipping duplicate tab: 'Dup'" in out


def test_generation_ui_returns_namespaced_components(make_ext, monkeypatch):
    cfg = make_ext({
        'fx.py': (
            "import gradio as gr\n"
            "def render(tab):\n"
            "    return {'on': gr.Checkbox(label=tab), 'junk': 5}\n"
            "def register(api): api.add_ui(render, tabs='txt2img')\n"
        ),
        'broken.py': (
            "def render(tab): raise RuntimeError('ui')\n"
            "def register(api): api.add_ui(render)\n"
        ),
    })
    monkeypatch.setattr(
        'modules.extensions.ui.ensure_loaded', lambda: load_extensions(cfg))
    with gr.Blocks():
        inputs = create_extensions_ui('txt2img')
        assert list(inputs) == ['ext_fx_on']
        assert create_extensions_ui('imgedit') == {}


def test_info_tab_only_when_extensions_exist(make_ext, monkeypatch):
    cfg = make_ext({})
    monkeypatch.setattr(
        'modules.extensions.ui.ensure_loaded', lambda: load_extensions(cfg))
    with gr.Blocks() as demo, gr.Tabs():
        render_extensions_info()
    assert not any(
        getattr(b, 'label', None) == 'Extensions'
        for b in demo.blocks.values())


def test_info_rows_escape_and_show_everything(make_ext):
    from modules.extensions.ui import _info_rows
    cfg = make_ext({'x.py': (
        "NAME = '<b>X</b>'\nVERSION = '2'\nAUTHOR = 'A'\n"
        "URL = 'javascript:alert(1)'\nDESCRIPTION = 'd'\n"
        "REQUIRES = ['pillow>=1']\nraise OSError('boom')\n"
    ), 'y.py': (
        "URL = 'https://example.com'\ndef register(api): pass\n"
    )})
    load_extensions(cfg)
    rows = _info_rows()
    assert '<b>X</b>' not in rows and '&lt;b&gt;X&lt;/b&gt;' in rows
    assert '<a href="javascript' not in rows
    assert 'href="https://example.com"' in rows
    assert 'pillow&gt;=1' in rows
    assert 'failed' in rows and 'boom' in rows

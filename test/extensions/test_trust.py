"""Asking before running a new or changed extension."""

import os

from modules.extensions import trust
from modules.extensions.host import extensions
from modules.extensions.loader import load_extensions

SOURCE = (
    "NAME = 'Thing'\nVERSION = '2.0'\nHOSTS = ['example.com']\n"
    "def register(api): pass\n"
)


def _restart():
    """Forget what was loaded, as a new start of the app would."""
    from modules.extensions import host, loader
    host._reset()
    loader._reset()


def answers(monkeypatch, *replies):
    """Answer the dialog; returns the texts it showed."""
    shown, replies = [], list(replies)
    monkeypatch.setattr(
        trust, 'ask', lambda text: shown.append(text) or replies.pop(0))
    return shown


def test_a_new_extension_shows_what_it_declares_and_where_it_saves(
    make_ext, monkeypatch
):
    shown = answers(monkeypatch, True)
    assert load_extensions(make_ext({'thing.py': SOURCE})) == ['thing']
    (text,) = shown
    assert 'New extension: Thing 2.0' in text
    assert 'example.com' in text and 'not enforced' in text
    assert 'If it saves any output' in text
    assert 'outputs/extensions/thing/' in text
    assert 'no sandbox' in text


def test_a_rejected_extension_is_not_loaded_and_asks_again(
    make_ext, monkeypatch
):
    cfg = make_ext({'thing.py': SOURCE})
    shown = answers(monkeypatch, False, True)
    assert load_extensions(cfg) == []
    (ext,) = extensions()
    assert ext.status == 'skipped' and 'ask again' in ext.reason
    assert os.path.isfile(cfg.data['ext_dir'] + '/thing.py')   # kept
    _restart()
    assert load_extensions(cfg) == ['thing']            # asked again
    assert len(shown) == 2


def test_an_approved_extension_is_not_asked_again_until_it_changes(
    make_ext, monkeypatch, tmp_path
):
    cfg = make_ext({'thing.py': SOURCE})
    shown = answers(monkeypatch, True, True)
    load_extensions(cfg)
    _restart()
    assert load_extensions(cfg) == ['thing']
    assert len(shown) == 1                              # not asked again

    path = tmp_path / 'exts' / 'thing.py'
    path.write_text('API_VERSION = 1\n' + SOURCE + '# updated\n')
    _restart()
    assert load_extensions(cfg) == ['thing']
    assert len(shown) == 2 and 'Updated extension: Thing' in shown[1]


def test_a_change_in_any_file_of_a_folder_extension_counts(
    make_ext, monkeypatch, tmp_path
):
    cfg = make_ext({'d/extension.py': SOURCE, 'd/helper.py': 'X = 1\n'})
    shown = answers(monkeypatch, True, True)
    load_extensions(cfg)
    (tmp_path / 'exts' / 'd' / 'helper.py').write_text('X = 2\n')
    _restart()
    load_extensions(cfg)
    assert len(shown) == 2


def test_without_a_terminal_nothing_new_loads(make_ext, monkeypatch):
    monkeypatch.setattr(trust, 'interactive', lambda: False)
    monkeypatch.setattr(trust, 'ask', lambda text: (_ for _ in ()).throw(
        AssertionError('asked without a terminal')))
    assert load_extensions(make_ext({'thing.py': SOURCE})) == []
    assert 'there is none' in extensions()[0].reason


def test_nothing_runs_before_the_answer(make_ext, monkeypatch, tmp_path):
    marker = tmp_path / 'ran'
    answers(monkeypatch, False)
    cfg = make_ext({'evil.py': f"open({str(marker)!r}, 'w')\n"
                               "def register(api): pass\n"})
    load_extensions(cfg)
    assert not marker.exists()


def test_the_approval_of_a_removed_extension_is_forgotten(
    make_ext, monkeypatch, tmp_path
):
    import json

    cfg = make_ext({'thing.py': SOURCE, 'other.py': SOURCE})
    shown = answers(monkeypatch, True, True, True)
    load_extensions(cfg)
    store = trust.store_path()
    assert set(json.load(open(store))) == {'thing', 'other'}

    (tmp_path / 'exts' / 'other.py').unlink()
    _restart()
    assert load_extensions(cfg) == ['thing']
    assert set(json.load(open(store))) == {'thing'}

    (tmp_path / 'exts' / 'other.py').write_text(
        'API_VERSION = 1\n' + SOURCE)                     # put back
    _restart()
    load_extensions(cfg)
    assert len(shown) == 3                                # asked again


def test_an_unreadable_folder_forgets_nothing(
    make_ext, monkeypatch, tmp_path
):
    import json

    cfg = make_ext({'thing.py': SOURCE})
    answers(monkeypatch, True)
    load_extensions(cfg)
    store = trust.store_path()

    def broken(path):
        raise PermissionError('no')

    monkeypatch.setattr(os, 'listdir', broken)
    _restart()
    assert load_extensions(cfg) == []
    monkeypatch.undo()
    assert set(json.load(open(store))) == {'thing'}


def test_the_dialog_shows_the_reasons_and_says_when_there_is_none(
    make_ext, monkeypatch
):
    shown = answers(monkeypatch, True)
    load_extensions(make_ext({'why.py': (
        "HOSTS = {'a.example.com': 'the search', 'b.example.com': ''}\n"
        "REQUIRES = {'pillow>=10': 'reads the images'}\n"
        "def register(api): pass\n")}))
    (text,) = shown
    assert '- a.example.com: the search' in text
    assert '- b.example.com: (no reason given)' in text
    assert '- pillow>=10: reads the images' in text


def test_the_dialog_marks_entries_written_as_a_plain_list(
    make_ext, monkeypatch
):
    shown = answers(monkeypatch, True)
    load_extensions(make_ext({'plain.py': (
        "HOSTS = ['a.example.com']\nREQUIRES = ['pillow>=10']\n"
        "def register(api): pass\n")}))
    assert shown[0].count('(no reason given)') == 2


def test_the_dialog_shows_the_full_sha256_and_no_wrapped_lines(
    make_ext, monkeypatch
):
    shown = answers(monkeypatch, True)
    cfg = make_ext({'thing.py': SOURCE})
    load_extensions(cfg)
    (text,) = shown
    (sha,) = [line.split()[-1] for line in text.splitlines()
              if line.strip().startswith('SHA256:')]
    assert len(sha) == 64 and '...' not in text
    # a sentence is one line: the warning and the download folder
    assert any('no sandbox: it can do more than it declares' in line
               for line in text.splitlines())

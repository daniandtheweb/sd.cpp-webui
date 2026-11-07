import sys

import pytest
from packaging.requirements import Requirement

from modules.extensions import deps
from modules.extensions.host import extensions
from modules.extensions.loader import load_extensions


YES = lambda reqs: True  # noqa: E731


def req(text):
    return Requirement(text)


def test_version_specifier():
    assert not deps.is_satisfied(req('packaging>=9999'))


def test_missing_package():
    assert not deps.is_satisfied(req('sdcpp-fake-pkg'))


def test_extra_requirements_are_checked(monkeypatch):
    # a plain install of a package whose [cpu] extra adds a missing dep
    # must not count as satisfied (rembg without onnxruntime)
    class Dist:
        version = '2.0'
        requires = ['sdcpp-fake-backend; extra == "cpu"', 'packaging']

    real = deps.importlib.metadata.distribution

    def fake(name):
        return Dist() if name == 'sdcpp-fake-lib' else real(name)

    monkeypatch.setattr(deps.importlib.metadata, 'distribution', fake)
    assert deps.is_satisfied(req('sdcpp-fake-lib'))
    assert not deps.is_satisfied(req('sdcpp-fake-lib[cpu]'))


@pytest.mark.parametrize('entry', [
    '--index-url=https://evil/simple',
    '',
])
def test_rejects_non_plain_entries(entry):
    ok, bad = deps.parse_requires([entry])
    assert ok == [] and bad == [entry]


def test_ensure_installs_then_rechecks(monkeypatch):
    state = {'installed': False}
    monkeypatch.setattr(
        deps, 'is_satisfied', lambda r, seen=frozenset(): state['installed'])

    def fake_install(reqs):
        state['installed'] = True
        state['reqs'] = [str(r) for r in reqs]

    monkeypatch.setattr(deps, 'install', fake_install)
    assert deps.ensure(['somepkg>=1'], YES) is None
    assert state['reqs'] == ['somepkg>=1']


def test_ensure_reports_failed_install(monkeypatch):
    monkeypatch.setattr(deps, 'install', lambda reqs: 'installer exited 1')
    assert deps.ensure(['sdcpp-fake-pkg>=1'], YES) == 'installer exited 1'


def test_ensure_reports_still_missing(monkeypatch):
    monkeypatch.setattr(deps, 'install', lambda reqs: None)
    assert 'still missing' in deps.ensure(['sdcpp-fake-pkg>=1'], YES)


def test_installer_prefers_pip_then_uv(monkeypatch):
    monkeypatch.setattr(deps.importlib.util, 'find_spec', lambda n: object())
    assert deps._installer(['a'])[1:4] == ['-m', 'pip', 'install']

    monkeypatch.setattr(deps.importlib.util, 'find_spec', lambda n: None)
    monkeypatch.setattr(deps.shutil, 'which', lambda n: '/bin/uv')
    assert deps._installer(['a']) == [
        '/bin/uv', 'pip', 'install', '--python', sys.executable, 'a']


def test_loader_installs_missing_requires_when_confirmed(
    make_ext, monkeypatch, tmp_path
):
    calls = []

    def fake_install(reqs):
        calls.append([str(r) for r in reqs])
        (tmp_path / 'sdcpp_fake_pkg.py').write_text('')
        info = tmp_path / 'sdcpp_fake_pkg-1.0.dist-info'
        info.mkdir()
        (info / 'METADATA').write_text(
            'Metadata-Version: 2.1\nName: sdcpp_fake_pkg\nVersion: 1.0\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        deps.importlib.invalidate_caches()
        return None

    monkeypatch.setattr(deps, 'install', fake_install)
    monkeypatch.setattr(deps, 'ask_install', lambda name, reqs, reasons=None: True)
    cfg = make_ext({'needs.py': (
        "REQUIRES = ['sdcpp_fake_pkg>=1']\n"
        "def register(api): pass\n"
    )})
    assert load_extensions(cfg) == ['needs']
    assert calls == [['sdcpp_fake_pkg>=1']]


def test_loader_skips_when_install_fails(make_ext, monkeypatch, capsys):
    monkeypatch.setattr(deps, 'install', lambda reqs: 'installer exited 1')
    monkeypatch.setattr(deps, 'ask_install', lambda name, reqs, reasons=None: True)
    cfg = make_ext({
        'needs.py': "REQUIRES = ['sdcpp_fake_pkg>=1']\ndef register(api): pass\n",
        'ok.py': "def register(api): pass\n",
    })
    assert load_extensions(cfg) == ['ok']
    needs = next(e for e in extensions() if e.id == 'needs')
    assert needs.status == 'skipped'
    assert 'installer exited 1' in needs.reason


def test_loader_rejects_invalid_requires_before_installing(
    make_ext, monkeypatch
):
    monkeypatch.setattr(deps, 'install', lambda reqs: pytest.fail('installed'))
    cfg = make_ext({'evil.py': (
        "REQUIRES = ['--index-url=https://evil/simple']\n"
        "def register(api): pass\n"
    )})
    assert load_extensions(cfg) == []
    assert 'invalid REQUIRES' in extensions()[0].reason


def test_nothing_is_installed_unconfirmed(make_ext, monkeypatch):
    monkeypatch.setattr(deps, 'install', lambda reqs: pytest.fail('installed'))
    monkeypatch.setattr(deps, 'ask_install', lambda name, reqs, reasons=None: False)
    cfg = make_ext({
        'needs.py': "REQUIRES = ['sdcpp_fake_pkg>=1']\ndef register(api): pass\n",
        'ok.py': "def register(api): pass\n",
    })
    assert load_extensions(cfg) == ['ok']
    needs = next(e for e in extensions() if e.id == 'needs')
    assert needs.status == 'skipped'
    assert 'not installed (sdcpp_fake_pkg>=1)' in needs.reason


class _Stdin:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def test_ask_install_answers(monkeypatch):
    reqs = [req('pkg>=1')]
    monkeypatch.setattr(sys, 'stdin', _Stdin(True))
    for typed, expected in [('y', True), ('YES', True), ('', False),
                            ('n', False), ('maybe', False)]:
        monkeypatch.setattr('builtins.input', lambda prompt, t=typed: t)
        assert deps.ask_install('X', reqs) is expected


def test_ask_install_prompt_names_the_packages(monkeypatch):
    seen = []
    monkeypatch.setattr(sys, 'stdin', _Stdin(True))
    monkeypatch.setattr('builtins.input', lambda p: seen.append(p) or 'n')
    deps.ask_install('Masker', [req('rembg[cpu]')])
    assert seen == [
        "Extension 'Masker' requires missing package(s): rembg[cpu]. "
        "Install now? [y/N] "]


def test_ask_install_is_no_without_a_terminal(monkeypatch):
    monkeypatch.setattr(sys, 'stdin', _Stdin(False))
    monkeypatch.setattr(
        'builtins.input', lambda p: pytest.fail('asked without a terminal'))
    assert deps.ask_install('X', [req('pkg')]) is False

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr(sys, 'stdin', _Stdin(True))
    monkeypatch.setattr('builtins.input', eof)
    assert deps.ask_install('X', [req('pkg')]) is False


def test_ask_install_warns_about_the_risk_first(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', _Stdin(True))
    monkeypatch.setattr('builtins.input', lambda p: 'n')
    deps.ask_install('X', [req('pkg')])
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'no sandbox' in out


def test_ask_install_shows_why_a_package_is_needed(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'stdin', _Stdin(True))
    monkeypatch.setattr('builtins.input', lambda p: 'n')
    deps.ask_install('X', [req('rembg[cpu]'), req('other')],
                     {'rembg[cpu]': 'finds the foreground', 'other': ''})
    out = capsys.readouterr().out
    assert 'rembg: finds the foreground' in out
    assert 'other:' not in out


# ----------------------------------------------- every package has a version

@pytest.mark.parametrize('entry', [
    'pillow',
    'pkg!=1.0',
    'not a requirement',
])
def test_a_package_without_a_version_is_rejected(entry):
    ok, bad = deps.parse_requires([entry])
    assert (ok, bad) == ([], [entry])


@pytest.mark.parametrize('entry', [
    'pillow>=10',
    'pillow~=10.1',
    'pillow>=10,<11',
    'pillow>=10,!=10.2',
])
def test_a_minimum_a_maximum_or_a_pin_is_accepted(entry):
    ok, bad = deps.parse_requires([entry])
    assert [str(r) for r in ok] == [str(req(entry))] and bad == []


def test_an_extension_without_versions_fails_before_it_is_asked_about(
    make_ext, monkeypatch
):
    from modules.extensions import trust
    monkeypatch.setattr(trust, 'ask', lambda text: pytest.fail('asked'))
    cfg = make_ext({'loose.py': (
        "REQUIRES = ['pillow']\ndef register(api): pass\n")})
    assert load_extensions(cfg) == []
    (ext,) = extensions()
    assert ext.status == 'failed'
    assert 'every package needs a version' in ext.reason

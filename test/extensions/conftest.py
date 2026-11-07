import importlib
import sys
from types import SimpleNamespace

import pytest

# modules.extensions re-exports names that shadow submodules, so import
# the modules directly
ext_host = importlib.import_module('modules.extensions.host')
ext_loader = importlib.import_module('modules.extensions.loader')


@pytest.fixture(autouse=True)
def reset_extensions(monkeypatch, tmp_path_factory):
    monkeypatch.setattr(ext_loader, 'ENABLED', True)
    # Approve everything, into a file of the test: never the real one
    store = str(tmp_path_factory.mktemp('trust') / 'trusted.json')
    monkeypatch.setattr(ext_loader.trust, 'store_path', lambda: store)
    monkeypatch.setattr(ext_loader.trust, 'interactive', lambda: True)
    monkeypatch.setattr(ext_loader.trust, 'ask', lambda text: True)
    def clean():
        ext_host._reset()
        ext_loader._reset()
        for key in [k for k in sys.modules if k.startswith('sdcpp_ext_')]:
            del sys.modules[key]

    clean()
    yield
    clean()


@pytest.fixture()
def make_ext(tmp_path):
    """
    Create an extensions folder with the given {path: source} files and
    return a config-like object pointing at it. The entry files declare
    API_VERSION = 1 (mandatory) unless they have one or declare_api is off.
    """
    def make(files: dict[str, str], declare_api: bool = True):
        root = tmp_path / 'exts'
        root.mkdir(exist_ok=True)
        for rel, source in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            is_entry = rel.count('/') == 0 or rel.endswith('/extension.py')
            if declare_api and is_entry and 'API_VERSION' not in source:
                source = 'API_VERSION = 1\n' + source
            path.write_text(source)
        return SimpleNamespace(data={'ext_dir': str(root)})

    return make


@pytest.fixture()
def config():
    from modules.shared_instance import config
    return config

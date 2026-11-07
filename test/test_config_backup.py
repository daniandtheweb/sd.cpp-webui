import json

import pytest

from modules.settings import ConfigManager
from modules.settings.store import _strip_trailing_commas


def _load(tmp_path, monkeypatch, text):
    path = tmp_path / 'config.json'
    path.write_text(text)
    monkeypatch.setenv('SD_WEBUI_CONFIG_PATH', str(path))
    return ConfigManager(str(path)), path


def test_trailing_comma_is_repaired_and_settings_survive(
    tmp_path, capsys, monkeypatch
):
    text = '{"def_theme": "mine", "nested": {"a": [1, 2,],},\n  "x": 1,\n}'
    manager, path = _load(tmp_path, monkeypatch, text)

    assert manager.data['def_theme'] == 'mine'
    assert manager.data['nested'] == {'a': [1, 2]}
    assert manager.data['x'] == 1
    assert (tmp_path / 'config.json.broken').read_text() == text
    assert 'removed the trailing comma' in capsys.readouterr().out
    json.loads(path.read_text())  # saved back as valid JSON


def test_unrepairable_config_is_backed_up_before_defaults_overwrite_it(
    tmp_path, capsys, monkeypatch
):
    text = '{"def_theme": "mine" "x": 1}'  # missing comma
    manager, _ = _load(tmp_path, monkeypatch, text)

    assert (tmp_path / 'config.json.broken').read_text() == text
    assert 'Using empty configuration' in capsys.readouterr().out
    assert 'def_theme' in manager.data  # only the defaults


@pytest.mark.parametrize('text, expected', [
    ('{"a": 1,}', '{"a": 1}'),
    ('[1, 2 , ]', '[1, 2  ]'),
    ('{"a": ",}", "b": "x\\",]"}', '{"a": ",}", "b": "x\\",]"}'),
    ('{"a": [1,\n 2,\n],\n}', '{"a": [1,\n 2\n]\n}'),
])
def test_strip_trailing_commas(text, expected):
    assert _strip_trailing_commas(text) == expected

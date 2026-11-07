import json
from types import SimpleNamespace

import pytest

from modules.interfaces.common import models_tab
from modules.shared_instance import config
from modules.utils import model_files


@pytest.fixture()
def folder(tmp_path, monkeypatch):
    models = tmp_path / 'checkpoints'
    (models / 'sub').mkdir(parents=True)
    (models / 'a.safetensors').write_bytes(b'x' * 2048)
    (models / 'sub' / 'b.gguf').write_bytes(b'y')
    (models / 'notes.txt').write_text('not a model')
    (tmp_path / 'outside.safetensors').write_bytes(b'z')
    monkeypatch.setitem(config.data, 'ckpt_dir', str(models))
    return models


def test_the_list_has_every_model_with_text_and_numbers(folder):
    models = model_files.list_models('Checkpoint')
    assert [m['name'] for m in models] == ['a.safetensors', 'sub/b.gguf']
    assert models[0]['size'] == '2.0 KB' and models[0]['bytes'] == 2048
    assert models[1]['size'] == '1 B' and len(models[0]['modified']) == 16
    assert model_files.list_models('Nope') == []


def test_rename_keeps_the_extension_and_the_subfolder(folder):
    message, new = model_files.rename_model('Checkpoint', 'sub/b.gguf', 'c')
    assert (message, new) == ('Renamed to c.gguf.', 'sub/c.gguf')
    assert (folder / 'sub' / 'c.gguf').is_file()
    assert model_files.rename_model(
        'Checkpoint', 'a.safetensors', 'new.safetensors')[1] == (
        'new.safetensors')


@pytest.mark.parametrize('new', ['', '../x', '.hidden'])
def test_rename_refuses_names_that_are_not_plain(folder, new):
    message, renamed = model_files.rename_model(
        'Checkpoint', 'a.safetensors', new)
    assert renamed is None and (folder / 'a.safetensors').is_file()


def test_rename_never_replaces_a_file(folder):
    (folder / 'taken.safetensors').write_bytes(b'mine')
    message, renamed = model_files.rename_model(
        'Checkpoint', 'a.safetensors', 'taken')
    assert renamed is None and 'already exists' in message
    assert (folder / 'taken.safetensors').read_bytes() == b'mine'


@pytest.mark.parametrize('name', ['../outside.safetensors', 'notes.txt'])
def test_only_model_files_inside_the_folder_can_be_changed(folder, name):
    assert model_files.rename_model('Checkpoint', name, 'x')[1] is None
    assert model_files.remove_model('Checkpoint', name)[1] is False
    assert (folder.parent / 'outside.safetensors').is_file()
    assert (folder / 'notes.txt').is_file()


def test_remove_deletes_only_that_file(folder):
    assert model_files.remove_model('Checkpoint', 'sub/b.gguf') == (
        'Removed b.gguf.', True)
    assert not (folder / 'sub' / 'b.gguf').exists()
    assert (folder / 'a.safetensors').is_file()


def _click(**event):
    return SimpleNamespace(**event)


def _value(raw):
    return json.loads(raw)


def test_the_list_value_has_the_models_and_the_message(folder, monkeypatch):
    value = _value(models_tab.show_models('Checkpoint'))
    assert [m['name'] for m in value['models']] == [
        'a.safetensors', 'sub/b.gguf']
    assert value['message'] == '' and value['folder'] == str(folder)
    monkeypatch.setitem(config.data, 'vae_dir', str(folder / 'nowhere'))
    missing = _value(models_tab.show_models('VAE'))
    assert missing['ok'] is False and 'does not exist' in missing['message']


def test_a_click_on_rename_renames_and_says_so(folder):
    value = _value(models_tab.change_model('Checkpoint', _click(
        action='rename', name='a.safetensors', new_name='renamed')))
    assert value['ok'] and 'Renamed to renamed.safetensors' in value['message']
    assert [m['name'] for m in value['models']][0] == 'renamed.safetensors'


def test_a_click_on_remove_deletes_and_a_bad_one_is_an_error(folder):
    value = _value(models_tab.change_model('Checkpoint', _click(
        action='remove', name='a.safetensors')))
    assert value['ok'] and not (folder / 'a.safetensors').exists()
    value = _value(models_tab.change_model('Checkpoint', _click(
        action='remove', name='../outside.safetensors')))
    assert value['ok'] is False
    assert (folder.parent / 'outside.safetensors').is_file()


def test_an_unknown_click_changes_nothing(folder):
    value = _value(models_tab.change_model('Checkpoint', _click(action='x')))
    assert len(value['models']) == 2

"""The directories --allow-insecure-dir lets Gradio serve."""

import os

from sdcpp_webui import get_allowed_paths


def test_external_dirs_are_allowed(tmp_path):
    app = tmp_path / 'app'
    app.mkdir()
    outside = tmp_path / 'models'
    outside.mkdir()
    allowed = get_allowed_paths({'ckpt_dir': str(outside)}, str(app))
    assert allowed == [str(outside)]


def test_dirs_inside_the_app_need_no_permission(tmp_path):
    app = tmp_path / 'app'
    (app / 'models').mkdir(parents=True)
    assert get_allowed_paths({'ckpt_dir': str(app / 'models')}, str(app)) == []


def test_the_extensions_folder_is_never_served(tmp_path):
    """It holds code, so even outside the app it is not offered."""
    app = tmp_path / 'app'
    app.mkdir()
    elsewhere = tmp_path / 'my_extensions'
    elsewhere.mkdir()
    link = tmp_path / 'linked'
    os.symlink(elsewhere, link)
    for folder in (elsewhere, link):
        assert get_allowed_paths({'ext_dir': str(folder)}, str(app)) == []
    both = get_allowed_paths(
        {'ext_dir': str(elsewhere), 'ckpt_dir': str(tmp_path)}, str(app))
    assert str(elsewhere) not in both and str(tmp_path) in both

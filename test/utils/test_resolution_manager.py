import os

import pytest

from modules.utils.resolution_manager import (
    ResolutionManager, BUILTIN_RESOLUTIONS
)


@pytest.fixture
def manager(tmp_path):
    return ResolutionManager(tmp_path / "resolutions.json")


def test_file_is_initialized(manager):
    assert os.path.isfile(manager.resolutions_path)


def test_builtin_resolutions_always_present(manager):
    names = manager.get_resolutions()
    for name in BUILTIN_RESOLUTIONS:
        assert name in names


def test_get_resolution_returns_builtin(manager):
    assert manager.get_resolution("512x512") == (512, 512)


def test_get_resolution_returns_none_when_missing(manager):
    assert manager.get_resolution("missing") is None


def test_add_resolution_persists_to_disk(manager):
    assert manager.add_resolution("myres", 100, 200) is True
    assert manager.get_resolution("myres") == (100, 200)

    reloaded = ResolutionManager(manager.resolutions_path)
    assert reloaded.get_resolution("myres") == (100, 200)


def test_add_resolution_refuses_builtin_name(manager):
    for name in BUILTIN_RESOLUTIONS:
        assert manager.add_resolution(name, 1, 1) is False


def test_add_resolution_refuses_existing_name(manager):
    assert manager.add_resolution("dup", 100, 100) is True
    assert manager.add_resolution("dup", 200, 200) is False


def test_add_resolution_casts_floats_to_int(manager):
    # Gradio sliders can hand back floats (e.g. 960.0)
    assert manager.add_resolution("floats", 960.0, 1280.0) is True
    assert manager.get_resolution("floats") == (960, 1280)


def test_add_resolution_refuses_empty_name(manager):
    assert manager.add_resolution("", 100, 100) is False
    assert manager.add_resolution("   ", 100, 100) is False


def test_delete_resolution_removes_user_preset(manager):
    assert manager.add_resolution("gone", 100, 100) is True
    assert manager.delete_resolution("gone") is True
    assert manager.get_resolution("gone") is None
    assert "gone" not in manager.get_resolutions()


def test_delete_resolution_refuses_builtin(manager):
    for name in BUILTIN_RESOLUTIONS:
        assert manager.delete_resolution(name) is False
    # Built-ins survive and file is untouched
    assert manager.get_resolution("512x512") == (512, 512)


def test_delete_resolution_refuses_missing(manager):
    assert manager.delete_resolution("missing") is False

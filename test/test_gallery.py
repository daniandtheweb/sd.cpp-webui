def test_get_next_media_seq():
    from modules.gallery import get_next_media

    filename = get_next_media(subctrl=0)
    assert "1.png" == filename


def test_get_next_media_seq_conflict(app_root):
    from modules.gallery import get_next_media

    existing_file = app_root / "img2img" / "1.png"
    existing_file.touch()

    filename = get_next_media(subctrl=1)
    assert "2.png" == filename


import os
import time

import pytest
from PIL import Image


@pytest.fixture
def gallery_dir(tmp_path):
    """Points the txt2img gallery to an empty directory."""
    from modules.shared_instance import config

    saved = config.data.get('txt2img_dir')
    target = tmp_path / "txt2img"
    config.update_settings({'txt2img_dir': str(target)})
    yield target
    config.update_settings({'txt2img_dir': saved})


def _add_images(directory, names):
    directory.mkdir(exist_ok=True)
    for name in names:
        Image.new("RGB", (8, 8)).save(directory / name)
        # Distinct creation times for the date sorting
        time.sleep(0.01)


def _names(files):
    return [os.path.basename(f) for f in files]


def test_gallery_follows_config_dir(gallery_dir):
    """A directory missing at startup or changed later is still listed."""
    from modules.gallery import GalleryManager

    manager = GalleryManager()
    assert manager._get_sorted_files() == []

    _add_images(gallery_dir, ["1.png"])
    assert _names(manager._get_sorted_files()) == ["1.png"]


def test_gallery_name_sort_is_natural(gallery_dir):
    from modules.gallery import GalleryManager

    _add_images(gallery_dir, ["1.png", "10.png", "2.png", "2_1.png"])
    manager = GalleryManager()

    manager.sort_order = "Name (A-Z)"
    assert _names(manager._get_sorted_files()) == [
        "1.png", "2.png", "2_1.png", "10.png"
    ]
    manager.sort_order = "Name (Z-A)"
    assert _names(manager._get_sorted_files())[0] == "10.png"


def test_gallery_page_count(gallery_dir):
    from modules.gallery import GalleryManager, PAGE_SIZE

    _add_images(gallery_dir, [f"{i}.png" for i in range(PAGE_SIZE + 1)])
    manager = GalleryManager()

    _, page = manager.reload_gallery(page_num=99, ctrl_inp=0)
    assert page == 2


def test_gallery_selection_uses_shown_files(gallery_dir):
    """A new output must not shift what a click selects."""
    from modules.gallery import GalleryManager

    _add_images(gallery_dir, ["1.png", "2.png"])
    manager = GalleryManager()
    manager.reload_gallery(page_num=1, ctrl_inp=0,
                           sort_inp="Date (Newest First)")
    shown = manager.files[0]

    _add_images(gallery_dir, ["3.png"])

    class Selected:
        index = 0

    info = manager.get_media_info(Selected())
    assert info[9] == shown


def test_gallery_delete_removes_selected_file(gallery_dir):
    from modules.gallery import GalleryManager

    _add_images(gallery_dir, ["1.png", "2.png"])
    manager = GalleryManager()
    manager.reload_gallery(page_num=1, ctrl_inp=0,
                           sort_inp="Date (Newest First)")

    class Selected:
        index = 0

    manager.get_media_info(Selected())
    _add_images(gallery_dir, ["3.png"])

    manager.delete_media()
    assert _names(sorted(os.listdir(gallery_dir))) == ["1.png", "3.png"]


def test_get_next_media_counts_videos(tmp_path):
    from modules.shared_instance import config
    from modules.gallery import get_next_media

    saved = config.data.get('any2video_dir')
    target = tmp_path / "any2video"
    target.mkdir()
    for name in ("1.webm", "2.webm"):
        (target / name).touch()
    config.update_settings({'any2video_dir': str(target)})
    try:
        assert get_next_media(subctrl=3) == "3.webm"
    finally:
        config.update_settings({'any2video_dir': saved})

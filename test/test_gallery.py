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

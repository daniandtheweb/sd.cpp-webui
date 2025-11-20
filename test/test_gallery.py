def test_get_next_img_seq():
    from modules.gallery import get_next_img

    filename = get_next_img(subctrl=0)
    assert "1.png" == filename


def test_get_next_img_seq_conflict(app_root):
    from modules.gallery import get_next_img

    existing_file = app_root / "img2img" / "1.png"
    existing_file.touch()

    filename = get_next_img(subctrl=1)
    assert "2.png" == filename

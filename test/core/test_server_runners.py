import pytest

DIR_KEYS = ("txt2img_dir", "img2img_dir", "imgedit_dir")


@pytest.fixture(autouse=True)
def exe_name_mock(mocker):
    return mocker.patch(
        "modules.utils.sd_interface.exe_name", return_value="sd-cli"
    )


def test_api_runners_sequential_output_paths(app_root):
    from modules.shared_instance import config
    from modules.core.server import sdcpp_server

    dir_paths = {}
    for name in ("txt2img", "img2img", "imgedit"):
        dir_path = app_root / f"api_{name}"
        dir_path.mkdir(exist_ok=True)
        dir_paths[name] = dir_path

    saved = {key: config.data.get(key) for key in DIR_KEYS}

    try:
        config.update_settings(
            {
                "txt2img_dir": str(dir_paths["txt2img"]),
                "img2img_dir": str(dir_paths["img2img"]),
                "imgedit_dir": str(dir_paths["imgedit"]),
                "def_output_scheme": "Sequential",
            }
        )

        seed_files = {
            "txt2img": "5.png",
            "img2img": "10.png",
            "imgedit": "3.png",
        }
        for name, filename in seed_files.items():
            (dir_paths[name] / filename).touch()

        params = {"in_ip": "127.0.0.1", "in_port": "7860"}

        txt2img_runner = sdcpp_server.Txt2ImgApiRunner(params)
        txt2img_runner.prepare()

        img2img_runner = sdcpp_server.Img2ImgApiRunner(params)
        img2img_runner.prepare()

        imgedit_runner = sdcpp_server.ImgEditApiRunner(params)
        imgedit_runner.prepare()

        assert txt2img_runner.output_path == str(
            dir_paths["txt2img"] / "6.png"
        )
        assert img2img_runner.output_path == str(
            dir_paths["img2img"] / "11.png"
        )
        assert imgedit_runner.output_path == str(
            dir_paths["imgedit"] / "4.png"
        )
    finally:
        config.update_settings(saved)

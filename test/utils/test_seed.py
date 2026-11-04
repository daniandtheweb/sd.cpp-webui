"""Reuse last seed: what is remembered and what the button does."""
import pytest

from modules.utils import math_utils


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    monkeypatch.setattr(math_utils, '_last_seed', None)


def test_the_button_does_nothing_before_the_first_run():
    import gradio as gr
    assert math_utils.reuse_last_seed() == gr.skip()


def test_the_button_sets_the_seed_of_the_last_run():
    math_utils.remember_seed(1234)
    assert math_utils.reuse_last_seed()['value'] == 1234
    math_utils.remember_seed(7.0)
    assert math_utils.reuse_last_seed()['value'] == 7


@pytest.mark.parametrize('seed', [-1, None, '', 'x'])
def test_a_random_seed_is_not_remembered(seed):
    math_utils.remember_seed(42)
    math_utils.remember_seed(seed)
    assert math_utils.reuse_last_seed()['value'] == 42


def test_the_log_reports_the_seed_sdcpp_picked():
    from modules.utils.log_parser import LogParser
    parser = LogParser()
    for line in ("generating image: 1/3 - seed 555",
                 "generating image: 2/3 - seed 556",
                 "generating image: 3/3 - seed 557"):
        parser._parse_final_stats(line, {})
    assert math_utils.reuse_last_seed()['value'] == 555


def test_cli_run_remembers_its_seed(mocker, app_root):
    from modules.core.cli import sdcpp_cli
    mocker.patch.object(sdcpp_cli.subprocess_manager, 'run_subprocess',
                        lambda command, env=None: iter(()))
    runner = sdcpp_cli.Txt2ImgRunner(mode="img_gen", params={})
    runner.command = ['sd-cli', '-M', 'img_gen', '-s', '99']
    runner.fcommand = ''
    runner.outputs, runner.preview_path = [], None
    mocker.patch.object(runner, '_prepare_for_run')
    mocker.patch.object(runner, '_save_command')
    mocker.patch.object(runner, '_build_process_env', return_value={})
    next(runner.run())
    assert math_utils.reuse_last_seed()['value'] == 99


def test_server_request_remembers_its_seed():
    from modules.core.server.sdcpp_server import Txt2ImgApiRunner
    runner = Txt2ImgApiRunner({'in_ip': '127.0.0.1', 'in_port': 1,
                               'in_seed': 321})
    next(runner.run())
    assert math_utils.reuse_last_seed()['value'] == 321

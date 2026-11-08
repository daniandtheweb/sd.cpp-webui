"""CLI runner: stopping a generation."""

import gradio as gr
from PIL import Image


def _stopped_runner(mocker, outputs, preview_path):
    from modules.core.cli import sdcpp_cli

    def stopped_run(command, env=None):
        yield {"percent": 25, "step": 5, "total": 20, "status": "Speed"}
        sdcpp_cli.subprocess_manager.stopped = True

    mocker.patch.object(
        sdcpp_cli.subprocess_manager, 'run_subprocess', stopped_run)
    mocker.patch.object(sdcpp_cli.subprocess_manager, 'stopped', False)

    runner = sdcpp_cli.Txt2ImgRunner(mode="img_gen", params={})
    runner.fcommand = "sd-cli -M img_gen"
    runner.outputs = outputs
    runner.preview_path = preview_path
    mocker.patch.object(runner, '_prepare_for_run')
    mocker.patch.object(runner, '_save_command')
    mocker.patch.object(runner, '_build_process_env', return_value={})
    return runner


def test_cli_stop_keeps_the_command(mocker, app_root):
    """A stopped CLI run clears the statistics, not the command or images."""
    runner = _stopped_runner(mocker, [], None)

    command, progress, stats, images = list(runner.run())[-1]
    assert command == "sd-cli -M img_gen"
    assert progress['visible'] is False
    assert stats['visible'] is False and stats['value'] == ""
    assert images == gr.skip()          # nothing blanked: no step to keep


def test_cli_stop_keeps_the_last_step_saved_and_shown(mocker, tmp_path):
    """The preview of the last step is saved as <name>_stopped.png."""
    preview = tmp_path / "7_preview.png"
    Image.new('RGB', (8, 8), 'red').save(preview)
    out = str(tmp_path / "7.png")

    command, progress, stats, images = list(
        _stopped_runner(mocker, [out], str(preview)).run())[-1]
    assert images == [str(tmp_path / "7_stopped.png")]
    assert (tmp_path / "7_stopped.png").is_file() and not preview.exists()

    # A second stopped run of the same name does not replace it
    Image.new('RGB', (8, 8), 'blue').save(preview)
    images = list(_stopped_runner(mocker, [out], str(preview)).run())[-1][3]
    assert images == [str(tmp_path / "7_stopped_1.png")]
    assert (tmp_path / "7_stopped.png").is_file()

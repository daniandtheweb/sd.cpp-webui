"""CLI runner: stopping a generation."""


def test_cli_stop_keeps_the_command(mocker, app_root):
    """A stopped CLI run clears statistics and images, not the command."""
    from modules.core.cli import sdcpp_cli

    def stopped_run(command, env=None):
        yield {"percent": 25, "step": 5, "total": 20, "status": "Speed"}
        sdcpp_cli.subprocess_manager.stopped = True

    mocker.patch.object(
        sdcpp_cli.subprocess_manager, 'run_subprocess', stopped_run)
    mocker.patch.object(sdcpp_cli.subprocess_manager, 'stopped', False)

    runner = sdcpp_cli.Txt2ImgRunner(mode="img_gen", params={})
    runner.fcommand = "sd-cli -M img_gen"
    runner.outputs = []
    runner.preview_path = None
    mocker.patch.object(runner, '_prepare_for_run')
    mocker.patch.object(runner, '_save_command')
    mocker.patch.object(runner, '_build_process_env', return_value={})

    command, progress, stats, images = list(runner.run())[-1]
    assert command == "sd-cli -M img_gen"
    assert progress['visible'] is False
    assert stats['visible'] is False and stats['value'] == ""
    assert images['value'] is None

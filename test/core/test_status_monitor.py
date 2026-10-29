import httpx
import pytest


@pytest.fixture(autouse=True)
def reset_server_state():
    from modules.shared_instance import server_state

    saved = {
        'running': server_state.running,
        'latest_update': server_state.latest_update,
        'ip': getattr(server_state, 'ip', None),
        'port': getattr(server_state, 'port', None),
    }
    server_state.running = True
    server_state.latest_update = {}
    yield
    server_state.running = saved['running']
    server_state.latest_update = saved['latest_update']
    if 'ip' in saved:
        setattr(server_state, 'ip', saved['ip'])
    if 'port' in saved:
        setattr(server_state, 'port', saved['port'])


def test_get_active_model_name_offline_sentinel(mocker):
    """A connection failure must report 'Offline', not 'Loading weights...'."""
    import modules.core.server.status_monitor as sm

    mocker.patch.object(
        sm.httpx, 'get',
        side_effect=httpx.ConnectError('refused'),
    )
    assert sm.get_active_model_name('127.0.0.1', 7860) == 'Offline'


def test_wrapper_calls_get_active_model_name_once(mocker):
    """One HTTP probe per tick: the model name is fetched exactly once."""
    import modules.core.server.status_monitor as sm

    get_name = mocker.patch.object(
        sm, 'get_active_model_name', return_value='model.gguf'
    )
    mocker.patch.object(
        sm.queue_manager, 'is_running', return_value=False
    )

    status, btn, slider, text = sm.server_status_monitor_wrapper(
        '127.0.0.1', 7860
    )

    assert get_name.call_count == 1
    assert status == 'Running (Model: model.gguf)'
    assert btn['interactive'] is True


def test_wrapper_offline_status(mocker):
    """Offline sentinel maps to an unreachable status with a dead button."""
    import modules.core.server.status_monitor as sm

    mocker.patch.object(
        sm, 'get_active_model_name', return_value='Offline'
    )
    mocker.patch.object(
        sm.queue_manager, 'is_running', return_value=False
    )

    status, btn, slider, text = sm.server_status_monitor_wrapper(
        '127.0.0.1', 7860
    )

    assert status == 'Server unreachable (starting or crashed)'
    assert btn['interactive'] is False


def _make_runner():
    from modules.core.server import sdcpp_server

    runner = sdcpp_server.Txt2ImgApiRunner(
        {'in_ip': '127.0.0.1', 'in_port': 7860}
    )
    runner.prepare()
    return runner


def _drain_run(runner):
    """Exhausts the run() generator; returns the stats-slot update."""
    gen = runner.run()
    next(gen)  # first yield: fcommand + skips
    result = next(gen)  # second yield: stats slot
    return result


def test_api_runner_http_error_yielded(mocker):
    """Non-200 responses must surface an error in the stats slot (C9)."""
    from modules.core.server import sdcpp_server
    from modules.core.server.sdcpp_server import ApiTaskRunner

    mock_httpx = mocker.patch.object(sdcpp_server, 'httpx')
    resp = mocker.Mock()
    resp.status_code = 400
    resp.text = 'invalid lora path'
    mock_httpx.post.return_value = resp

    # Seed the lora cache: the error path must wipe it.
    ApiTaskRunner._lora_lookup_cache.clear()
    ApiTaskRunner._lora_lookup_cache['seed'] = '/some/lora.gguf'

    runner = _make_runner()
    result = _drain_run(runner)

    stats_update = result[2]
    assert stats_update['value'] == (
        'Error: server returned HTTP 400 invalid lora path'
    )
    assert ApiTaskRunner._lora_lookup_cache == {}


def test_api_runner_exception_yielded(mocker):
    """Unhandled exceptions must surface an error in the stats slot (C9)."""
    from modules.core.server import sdcpp_server
    from modules.core.server.sdcpp_server import ApiTaskRunner

    mock_httpx = mocker.patch.object(sdcpp_server, 'httpx')
    mock_httpx.post.side_effect = httpx.ConnectError('connection refused')

    ApiTaskRunner._lora_lookup_cache.clear()
    ApiTaskRunner._lora_lookup_cache['seed'] = '/some/lora.gguf'

    runner = _make_runner()
    result = _drain_run(runner)

    stats_update = result[2]
    assert stats_update['value'] == 'Error: ConnectError: connection refused'
    assert ApiTaskRunner._lora_lookup_cache == {}

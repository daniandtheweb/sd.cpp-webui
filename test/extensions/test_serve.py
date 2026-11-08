"""Restarting the sd-server with what an extension needs."""

import pytest

from modules.extensions import host
from modules.shared_instance import server_state


@pytest.fixture
def server(monkeypatch):
    calls = []
    monkeypatch.setattr(host, '_mode', lambda: 'server')
    monkeypatch.setattr(server_state, 'running', True)
    monkeypatch.setattr(server_state, 'launch_params', {
        'in_ckpt_model': "a.safetensors", 'in_cnnet': ""})
    import modules.core.server.manager as manager
    monkeypatch.setattr(manager, 'restart_server', calls.append)
    return calls


def run(steps):
    shown = []
    while True:
        try:
            shown.append(next(steps))
        except StopIteration as done:
            return shown, done.value


def test_nothing_changes_when_the_server_has_it(server):
    shown, before = run(host.serving({'in_ckpt_model': "a.safetensors"}))
    assert (shown, before, server) == ([], None, [])


def test_the_server_restarts_with_what_is_missing_and_is_restored(server):
    shown, before = run(host.serving({'in_cnnet': "tile.safetensors"}))
    assert shown and before == {'in_ckpt_model': "a.safetensors",
                                'in_cnnet': ""}
    assert server == [{'in_ckpt_model': "a.safetensors",
                       'in_cnnet': "tile.safetensors"}]
    host.restore_server(before)
    assert server[-1] == before
    host.restore_server(None)
    assert len(server) == 2


def test_a_server_that_fails_to_start_is_brought_back(monkeypatch, server):
    import modules.core.server.manager as manager
    tried = []

    def restart(params):
        tried.append(params)
        if params.get('in_cnnet'):
            raise RuntimeError("bad controlnet")
    monkeypatch.setattr(manager, 'restart_server', restart)
    with pytest.raises(RuntimeError, match="runs as before|as before"):
        run(host.serving({'in_cnnet': "tile.safetensors"}))
    assert tried[-1]['in_cnnet'] == ""


def test_cli_mode_does_nothing(monkeypatch):
    monkeypatch.setattr(host, '_mode', lambda: 'cli')
    assert run(host.serving({'in_cnnet': "x"})) == ([], None)

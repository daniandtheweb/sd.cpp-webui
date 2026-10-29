"""Generation stream: job updates survive a dropped browser stream."""

import threading

import gradio as gr


def _job(release):
    """A job that reports progress and waits for `release` to finish."""
    def fake_api(params):
        yield ("cmd", gr.update(visible=True, value=50),
               gr.update(visible=True, value="Speed: 1.00 it/s | ETA: 00:01"),
               gr.skip(), ["preview.png"])
        release.wait(5)
        yield ("cmd", gr.update(visible=False, value=0),
               gr.update(visible=False, value=""),
               gr.update(value="Total: 1.00s"), ["final.png"])
    return fake_api


def _next_until(stream, check):
    for snapshot in stream:
        if check(snapshot):
            return snapshot
    raise AssertionError("the stream ended first")


def test_stream_dropped_while_hidden_is_restored():
    """The page is hidden mid-generation and comes back after the end."""
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    stream = gen_stream.stream("session")
    _next_until(stream, lambda s: s[2].get('value', '').startswith("Speed"))
    # The browser drops the stream (hidden page), the job keeps running
    stream.close()
    release.set()

    # The page becomes visible again: the resync stream sends the end state
    snapshots = list(gen_stream.stream("session"))
    command, progress, status, stats, images, _ = snapshots[-1]
    assert progress['visible'] is False
    assert status['visible'] is False
    assert stats['value'] == "Total: 1.00s"
    assert images['value'] == ["final.png"]


def test_resync_resumes_a_running_job():
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    resync = gen_stream.stream("session")
    first = _next_until(resync, lambda s: s[1].get('value') == 50)
    assert first[1]['visible'] is True

    release.set()
    last = list(resync)[-1]
    assert last[3]['value'] == "Total: 1.00s"


def test_newest_stream_of_a_session_wins():
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    old = gen_stream.stream("a")
    other = gen_stream.stream("b")
    next(old)
    next(other)

    # A new stream of session "a" stops its old one, not session "b"
    new = gen_stream.stream("a")
    next(new)
    release.set()
    assert list(old) == []
    assert list(other)[-1][3]['value'] == "Total: 1.00s"
    assert list(new)[-1][3]['value'] == "Total: 1.00s"


def test_idle_resync_sends_nothing_new():
    from modules.utils.ui_events import GenerationStream

    def unused(params):
        yield from ()

    snapshots = list(GenerationStream(unused).stream("session"))
    assert len(snapshots) == 1
    assert all(update == gr.skip() for update in snapshots[0])

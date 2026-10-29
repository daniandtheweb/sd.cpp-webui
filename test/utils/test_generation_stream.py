"""Generation stream: job updates survive a dropped browser stream."""

import threading

import gradio as gr

from modules.ui.progress import progress_update


def _job(release):
    """A job that reports progress and waits for `release` to finish."""
    def fake_api(params):
        yield ("cmd",
               progress_update(50, 5, 10, "Speed: 1.00 it/s | ETA: 00:01"),
               gr.skip(), ["preview.png"])
        release.wait(5)
        yield ("cmd", progress_update(visible=False),
               gr.update(visible=True, value="Total: 1.00s"), ["final.png"])
    return fake_api


def _shown(snapshots):
    """What the browser shows after applying the snapshots in order."""
    state = [{} for _ in snapshots[0]]
    for snapshot in snapshots:
        for slot, update in enumerate(snapshot):
            state[slot].update(
                {k: v for k, v in update.items() if k != '__type__'}
            )
    return state


def _next_until(stream, check):
    """
    Reads the stream until a snapshot matches `check`. Returns all the
    snapshots read: unchanged outputs are skipped, so the state shown is
    the sum of them (see _shown).
    """
    read = []
    for snapshot in stream:
        read.append(snapshot)
        if check(snapshot):
            return read
    raise AssertionError("the stream ended first")


def test_stream_dropped_while_hidden_is_restored():
    """The page is hidden mid-generation and comes back after the end."""
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    stream = gen_stream.stream("session")
    _next_until(stream, lambda s: "Speed" in s[1].get('value', ''))
    # The browser drops the stream (hidden page), the job keeps running
    stream.close()
    release.set()

    # The page becomes visible again: the resync stream sends the end state
    snapshots = list(gen_stream.stream("session"))
    command, progress, stats, images, _ = _shown(snapshots)
    assert progress['visible'] is False
    assert stats['visible'] is True
    assert stats['value'] == "Total: 1.00s"
    assert images['value'] == ["final.png"]


def test_resync_resumes_a_running_job():
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    resync = gen_stream.stream("session")
    shown = _shown(
        _next_until(resync, lambda s: "Step 5/10" in s[1].get('value', ''))
    )
    assert shown[1]['visible'] is True
    # Statistics stay hidden while generating
    assert shown[2]['visible'] is False

    release.set()
    assert _shown(list(resync))[2]['value'] == "Total: 1.00s"


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
    assert _shown(list(other))[2]['value'] == "Total: 1.00s"
    assert _shown(list(new))[2]['value'] == "Total: 1.00s"


def test_idle_resync_sends_nothing_new():
    from modules.utils.ui_events import GenerationStream

    def unused(params):
        yield from ()

    snapshots = list(GenerationStream(unused).stream("session"))
    assert len(snapshots) == 1
    command, progress, stats, images, queue = snapshots[0]
    assert all(update == gr.skip() for update in (command, progress, stats, queue))
    # Only the images being clickable again
    assert images == gr.update(allow_preview=True, elem_classes=['gen-output'])


def test_images_not_clickable_while_generating():
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    stream = gen_stream.stream("session")
    preview = _shown(
        _next_until(stream, lambda s: s[3].get('value') == ["preview.png"])
    )
    assert preview[3]['allow_preview'] is False
    assert preview[3]['elem_classes'] == ['gen-output', 'gen-busy']

    release.set()
    final = _shown(list(stream))
    assert final[3]['value'] == ["final.png"]
    assert final[3]['allow_preview'] is True
    assert final[3]['elem_classes'] == ['gen-output']


def test_stop_keeps_the_command():
    """Stopping the queue clears the outputs but keeps the command."""
    import modules.utils.streaming_queue as queue_manager
    from modules.utils.ui_events import GenerationStream

    release = threading.Event()
    gen_stream = GenerationStream(_job(release))
    gen_stream.enqueue({})

    stream = gen_stream.stream("session")
    read = _next_until(stream, lambda s: "Speed" in s[1].get('value', ''))
    queue_manager.clear_all()
    release.set()

    command, progress, stats, images, _ = _shown(read + list(stream))
    assert command['value'] == "cmd"
    assert progress['visible'] is False
    assert stats == {'visible': False, 'value': ""}
    assert images['value'] is None


def test_outputs_are_sent_only_when_changed():
    """The command is sent once, not with every progress update."""
    from modules.utils.ui_events import GenerationStream

    def job(params):
        for step in range(1, 4):
            yield ("cmd", progress_update(step * 25, step, 4), gr.skip(),
                   gr.skip())
        yield ("cmd", progress_update(visible=False),
               gr.update(visible=True, value="Total: 1.00s"), ["final.png"])

    gen_stream = GenerationStream(job)
    gen_stream.enqueue({})
    snapshots = list(gen_stream.stream("session"))

    commands = [s[0] for s in snapshots if s[0] != gr.skip()]
    assert commands == [gr.update(value="cmd")]
    assert _shown(snapshots)[2]['value'] == "Total: 1.00s"

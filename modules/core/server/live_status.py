"""sd.cpp-webui - Live server status watcher

Pushes server lifecycle changes (Running / Stopped / model name) to the UI
without browser-side polling. A background thread polls the sd-server
process state and only broadcasts on actual transitions, so the browser
sees zero traffic while nothing changes.

Broadcast semantics: every subscribed status_stream() generator gets its
own queue, so multiple browser sessions stay in sync.
"""

import queue
import threading
import time

import gradio as gr

from modules.shared_instance import server_state
from modules.core.server.status_monitor import server_status_monitor_wrapper


_watcher_thread = None
_watcher_stop = None
_current_ip = None
_current_port = None

_subscribers = []
_sub_lock = threading.Lock()


def _subscribe():
    q = queue.Queue()
    with _sub_lock:
        _subscribers.append(q)
    return q


def _unsubscribe(q):
    with _sub_lock:
        if q in _subscribers:
            _subscribers.remove(q)


def _broadcast(item):
    with _sub_lock:
        targets = list(_subscribers)
    for q in targets:
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _final_status_message():
    exit_code = getattr(server_state, 'last_exit_code', None)
    if exit_code is None or exit_code == 0:
        return "Stopped (No Model Loaded)"
    return f"Crashed (exit code {exit_code})"


def _watch_loop(stop_event, ip, port):
    last = None
    startup_deadline = time.time() + 120

    while not stop_event.is_set():
        if not server_state.running:
            final_status = _final_status_message()
            if last != final_status:
                _broadcast((final_status, gr.update(interactive=False)))
            break

        try:
            status, btn, _, _ = server_status_monitor_wrapper(ip, port)
        except Exception:
            status = "Checking status..."
            btn = gr.update(interactive=False)

        if status != last:
            last = status
            _broadcast((status, btn))

        interval = 1.0 if time.time() < startup_deadline else 3.0

        if stop_event.wait(interval):
            break


def start_watcher(ip, port):
    global _watcher_thread, _watcher_stop, _current_ip, _current_port
    stop_watcher()

    _watcher_stop = threading.Event()
    _current_ip = ip
    _current_port = port
    _watcher_thread = threading.Thread(
        target=_watch_loop,
        args=(_watcher_stop, ip, port),
        daemon=True,
    )
    _watcher_thread.start()


def stop_watcher():
    global _watcher_thread
    if _watcher_stop is not None:
        _watcher_stop.set()
    if _watcher_thread is not None and _watcher_thread.is_alive():
        _watcher_thread.join(timeout=10)
    _watcher_thread = None

    # The loop exits without broadcasting when it's stopped, so tell
    # every tab the server is down
    if not server_state.running:
        _broadcast((_final_status_message(), gr.update(interactive=False)))


def status_stream():
    """Persistent generator drained by a Gradio .load() event.

    Yields (status, gen_btn_update, is_running) each time the watcher
    broadcasts a new state. is_running is read at yield time so it always
    reflects the current ground truth, even when the broadcast was queued
    a moment ago.
    """
    q = _subscribe()
    try:
        is_running = server_state.running
        if is_running and _current_ip is not None:
            try:
                status, btn, _, _ = server_status_monitor_wrapper(
                    _current_ip, _current_port
                )
                yield status, btn, is_running
            except Exception:
                yield "Running", gr.update(interactive=True), is_running
        else:
            yield _final_status_message(), gr.update(interactive=False), False

        while True:
            try:
                status, btn = q.get(timeout=30)
            except queue.Empty:
                continue
            yield status, btn, server_state.running
    finally:
        _unsubscribe(q)

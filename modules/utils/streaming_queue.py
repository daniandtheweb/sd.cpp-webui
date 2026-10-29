"""sd.cpp-webui - Streaming queue manager module

UI-mode companion to modules/utils/queue.py. The original queue caches the
latest snapshot of one job's state and relies on an external poller (UI
timer) to pull it into the interface. This variant pushes each generator
yield to a per-job subscriber queue that the click handler drains, which
lets Gradio stream updates natively without a timer.

CLI mode keeps using queue.py unchanged.
"""

import queue
import threading


class JobQueue:
    def __init__(self):
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._running = False
        self._current_owner = None
        self._current_sub = None
        self._worker = None

    def add_job(self, func, params, owner=None):
        sub = queue.Queue()
        self._jobs.put((func, params, owner, sub))
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._run, daemon=True
                )
                self._worker.start()
        return sub

    def clear_all(self):
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                break
            sub = job[3]
            try:
                sub.put_nowait(("stopped", None))
            except queue.Full:
                pass

        with self._lock:
            current_sub = self._current_sub
        if current_sub is not None:
            try:
                current_sub.put_nowait(("stopped", None))
            except queue.Full:
                pass

    def _run(self):
        while True:
            func, params, owner, sub = self._jobs.get()

            with self._lock:
                self._running = True
                self._current_owner = owner
                self._current_sub = sub

            try:
                for result in func(params):
                    # (command, progress, stats, images)
                    if isinstance(result, (tuple, list)) and len(result) >= 4:
                        normalized = tuple(result[:4])
                        sub.put(("update", normalized))
                    else:
                        print(
                            "Streaming worker warning: expected 4+ items "
                            f"from generator, got "
                            f"{len(result) if result else 0}"
                        )
            except Exception as exc:
                sub.put(("error", exc))
            else:
                sub.put(("done", None))
            finally:
                with self._lock:
                    self._running = False
                    self._current_owner = None
                    self._current_sub = None

    def get_queue_size(self):
        return self._jobs.qsize()

    def is_running(self):
        with self._lock:
            return self._running

    def current_owner(self):
        with self._lock:
            return self._current_owner


_queue = JobQueue()


def add_job(func, params, owner=None):
    return _queue.add_job(func, params, owner)


def clear_all():
    _queue.clear_all()


def get_queue_size():
    return _queue.get_queue_size()


def is_running():
    return _queue.is_running()


def current_owner():
    return _queue.current_owner()

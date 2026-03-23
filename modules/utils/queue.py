"""sd.cpp-webui - Queue manager module"""

import queue
import threading


_job_queue = queue.Queue()
_state_lock = threading.Lock()


def get_clean_state():
    return {
        "command": "",
        "progress": 0,
        "status": "Idle",
        "stats": "",
        "images": None,
        "is_running": False,
        "is_finished": False,
        "owner": None
    }


current_job_state = get_clean_state()


def _background_worker():
    """
    Generic Worker: Runs any function passed to it.
    Expected Function Signature: func(params) -> yields results
    """
    while True:
        job = _job_queue.get()

        func = job['func']
        params = job['params']
        owner = job.get('owner')

        with _state_lock:
            current_job_state.update(get_clean_state())
            current_job_state["is_running"] = True
            current_job_state["owner"] = owner

        try:
            for result in func(params):
                if isinstance(result, (tuple, list)) and len(result) >= 5:
                    (
                        current_job_state["command"],
                        current_job_state["progress"],
                        current_job_state["status"],
                        current_job_state["stats"],
                        current_job_state["images"],
                        *rest
                    ) = result
                else:
                    print(
                        "Worker Warning: Expected 5 items from generator, " +
                        f"got {len(result) if result else 0}"
                    )

        except Exception as e:
            with _state_lock:
                current_job_state["status"] = f"Error: {str(e)}"
            print(f"Worker Error: {e}")

        finally:
            with _state_lock:
                current_job_state["is_running"] = False
                current_job_state["is_finished"] = True
            _job_queue.task_done()


def start_worker():
    """Starts the background queue processor."""
    threading.Thread(target=_background_worker, daemon=True).start()


def add_job(func, params, owner=None):
    """
    Adds a generic job to the queue.
    :param func: The python function to run (e.g., txt2img)
    :param params: A dictionary of arguments for that function
    """
    job = {
        'func': func,
        'params': params,
        'owner': owner
    }
    _job_queue.put(job)


def get_queue_size():
    return _job_queue.qsize()


def get_status():
    with _state_lock:
        return current_job_state.copy()


def consume_finished():
    with _state_lock:
        if current_job_state["is_finished"]:
            current_job_state["is_finished"] = False
            return True
        else:
            return False

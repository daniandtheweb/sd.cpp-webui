"""sd.cpp-webui - Queue manager module"""

import queue
import threading


_job_queue = queue.Queue()


def get_clean_state():
    return {
        "command": "",
        "progress": 0,
        "status": "Idle",
        "stats": "",
        "images": None,
        "is_running": False,
        "is_finished": False
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

        current_job_state.update(get_clean_state())
        current_job_state["is_running"] = True

        try:
            for result in func(params):
                current_job_state["command"] = result[0]
                current_job_state["progress"] = result[1]
                current_job_state["status"] = result[2]
                current_job_state["stats"] = result[3]
                current_job_state["images"] = result[4]

        except Exception as e:
            current_job_state["status"] = f"Error: {str(e)}"
            print(f"Worker Error: {e}")

        finally:
            current_job_state["is_running"] = False
            current_job_state["is_finished"] = True
            _job_queue.task_done()


def start_worker():
    threading.Thread(target=_background_worker, daemon=True).start()


def add_job(func, params):
    """
    Adds a generic job to the queue.
    :param func: The python function to run (e.g., txt2img)
    :param params: A dictionary of arguments for that function
    """
    job = {
        'func': func,
        'params': params
    }
    _job_queue.put(job)


def get_queue_size():
    return _job_queue.qsize()


def get_status():
    return current_job_state

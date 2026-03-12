"""sd.cpp-webui - core - Server status monitor module"""

import requests

import gradio as gr

from modules.shared_instance import server_state
import modules.utils.queue as queue_manager


def get_server_status():
    """
    Check if the server is actually running.
    """
    if server_state.running:
        return "Running", gr.update(interactive=True)
    return "Stopped", gr.update(interactive=False)


def get_active_model_name(ip, port):
    """
    Gets the currently active model's name.
    """
    base_url = f"http://{ip}:{port}"

    try:
        resp = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=1.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                name = data[0].get("model_name") or data[0].get("title")
                if name and name.strip():
                    return name

        resp_opt = requests.get(f"{base_url}/sdapi/v1/options", timeout=1.0)
        if resp_opt.status_code == 200:
            opt_data = resp_opt.json()
            checkpoint = opt_data.get("sd_model_checkpoint")
            if checkpoint and checkpoint.strip():
                return checkpoint

        resp_v1 = requests.get(f"{base_url}/v1/models", timeout=1.0)
        if resp_v1.status_code == 200:
            v1_data = resp_v1.json()
            if "data" in v1_data and len(v1_data["data"]) > 0:
                return v1_data["data"][0].get("id", "Unknown")

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return "Loading weights..."
    except Exception:
        return "Checking status..."

    return "None"


def server_status_monitor_wrapper(ip, port):
    """
    Monitors server state and progress.
    Hides progress bar/status when progress reaches 100%.
    """
    server_state.ip = ip
    server_state.port = port

    if not server_state.running:
        return (
            "Stopped (No Model Loaded)",
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    model_name = get_active_model_name(ip, port)
    is_loading_model = model_name in ["Loading weights...", "Checking status..."]

    job = queue_manager.get_status()
    latest = server_state.latest_update

    is_active = job.get("is_running", False)

    if is_active:
        val = latest.get("percent", 0)
        status = latest.get("status", "")

        if val >= 100:
            slider_update = gr.update(visible=True, value=100)
            text_update = gr.update(visible=True, value="Decoding...")
        elif val <= 0:
            slider_update = gr.update(visible=True, value=0)
            text_update = gr.update(visible=True, value="Starting...")
        else:
            slider_update = gr.update(visible=True, value=val)
            text_update = gr.update(visible=True, value=status)

    elif is_loading_model:
        val = latest.get("percent", 0)
        slider_update = gr.update(visible=True, value=val)
        text_update = gr.update(visible=True, value="")

    else:
        slider_update = gr.update(visible=False)
        text_update = gr.update(visible=False)
        server_state.latest_update = {}

    model_name = get_active_model_name(ip, port)
    if is_loading_model or model_name == "None":
        combined_status = "Loading (Model initializing...)"
        btn_interactive = False
    else:
        combined_status = f"Running (Model: {model_name})"
        btn_interactive = True

    return (
        combined_status,
        gr.update(interactive=btn_interactive),
        slider_update,
        text_update
    )

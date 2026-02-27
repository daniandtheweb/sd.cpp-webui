"""sd.cpp-webui - core - Server status monitor module"""

import os
import requests

import gradio as gr

from modules.shared_instance import server_state, config


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
    Checks the internal state first, then pings the network.
    """
    if not server_state.running:
        return "Stopped (No Model Loaded)", gr.update(interactive=False)

    model_name = get_active_model_name(ip, port)

    if model_name in ["Loading weights...", "None", "Checking status..."]:
        combined_status = "Loading (Model initializing...)"
        btn_interactive = False
    else:
        combined_status = f"Running (Model: {model_name})"
        btn_interactive = True

    return combined_status, gr.update(interactive=btn_interactive)

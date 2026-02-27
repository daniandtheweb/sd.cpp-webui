"""sd.cpp-webui - core - Server status monitor module"""

import requests

import gradio as gr

from modules.shared_instance import server_state


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
    if not server_state.running:
        return (
            "Stopped (No Model Loaded)",
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    latest = server_state.latest_update
    slider_update = gr.update()
    text_update = gr.update()

    if "percent" in latest:
        val = latest["percent"]
        if val >= 100 or val == 0:
            slider_update = gr.update(visible=False, value=100)
            text_update = gr.update(visible=False, value="")
        else:
            slider_update = gr.update(visible=True, value=val)
            text_update = gr.update(visible=True, value=latest.get("status", ""))

    model_name = get_active_model_name(ip, port)
    if model_name in ["Loading weights...", "None", "Checking status..."]:
        combined_status = "Loading (Model initializing...)"
        btn_interactive = False
    else:
        combined_status = f"Running (Model: {model_name})"
        btn_interactive = True

    return combined_status, gr.update(interactive=btn_interactive), slider_update, text_update

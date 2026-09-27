"""sd.cpp-webui - UI component for deprecated compatibility options"""

import gradio as gr

from modules.shared_instance import config


def create_deprecated_ui():
    """
    Creates the 'Deprecated placement flags' accordion.
    """
    with gr.Accordion(label="Deprecated placement flags", open=False):
        clip_on_cpu = gr.Checkbox(
            label="Clip on CPU (--clip-on-cpu)",
            value=config.get('def_clip_on_cpu')
        )
        vae_on_cpu = gr.Checkbox(
            label="VAE on CPU (--vae-on-cpu)",
            value=config.get('def_vae_on_cpu')
        )
        control_net_cpu = gr.Checkbox(
            label="ControlNet on CPU (--control-net-cpu)",
            value=config.get('def_control_net_cpu')
        )

    return {
        'in_clip_on_cpu': clip_on_cpu,
        'in_vae_on_cpu': vae_on_cpu,
        'in_control_net_cpu': control_net_cpu,
    }

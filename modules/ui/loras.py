"""sd.cpp-webui - UI component for the lora selection"""

import gradio as gr

from modules.shared_instance import config, current_mode
from modules.ui.models import create_model_widget


def create_lora_sel_ui():
    """Create the lora selection UI"""
    with gr.Row():
        with gr.Accordion(
            label="LoRAs selector", open=False
        ):
            lora_model = create_model_widget(
                label="LoRA",
                dir_key='lora_dir',
                option_key=''
            )
            lora_prompt_switch = gr.Radio(
                label="Prompt",
                choices=["Positive", "Negative"],
                value="Positive"
            )
            lora_strength = gr.Slider(
                label="LoRA strength",
                value=config.get('def_lora_strength'),
                minimum=0.0,
                maximum=2.0
            )
            apply_lora_btn = gr.Button(
                value="Apply to prompt",
                variant='primary'
            )

    return {
        'in_lora_model': lora_model,
        'in_lora_prompt_switch': lora_prompt_switch,
        'in_lora_strength': lora_strength,
        'in_apply_lora_btn': apply_lora_btn,
    }

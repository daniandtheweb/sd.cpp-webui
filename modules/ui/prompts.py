"""sd.cpp-webui - UI component for the prompt saving feature"""

import gradio as gr

from modules.shared_instance import config
from .constants import RELOAD_SYMBOL


def create_prompts_ui(nprompt_support = True):
    """Create the prompts UI"""

    def save_and_refresh_prompts(name, p_prompt, n_prompt):
        config.add_prompt(name, p_prompt, n_prompt)
        return gr.update(choices=config.get_prompts(), value=name)

    def delete_and_refresh_prompts(name):
        config.delete_prompt(name)
        return gr.update(choices=config.get_prompts())

    def refresh_prompt_list():
        return gr.update(choices=config.get_prompts())

    with gr.Row():
        with gr.Accordion(
            label="Saved prompts", open=False
        ):
            with gr.Group():
                with gr.Column():
                    saved_prompts = gr.Dropdown(
                        label="Prompts",
                        choices=config.get_prompts(),
                        interactive=True,
                        allow_custom_value=False
                    )
                with gr.Column():
                    with gr.Row():
                        load_prompt_btn = gr.Button(
                            value="Load prompt", size="lg",
                        )
                        reload_prompts_btn = gr.Button(
                            value=RELOAD_SYMBOL
                        )
                    with gr.Row():
                        del_prompt_btn = gr.Button(
                            value="Delete prompt", size="lg",
                            variant="stop"
                        )
            with gr.Group():
                with gr.Column():
                    new_prompt = gr.Textbox(
                        label="New Prompt name",
                        placeholder="Prompt preset name"
                    )
                with gr.Column():
                    save_prompt_btn = gr.Button(
                        value="Save prompt", size="lg",
                    ) 
    with gr.Row():
        pprompt = gr.Textbox(
            placeholder="Positive prompt\nUse loras from the loras folder with: <lora:lora_name:lora_strenght>, for example: <lora:anime:0.8>",
            label="Positive Prompt",
            lines=3,
            show_copy_button=True,
            visible=True
        )
    with gr.Row():
        nprompt = gr.Textbox(
            placeholder="Negative prompt",
            label="Negative Prompt",
            lines=3,
            show_copy_button=True,
            visible=nprompt_support
        )

    save_prompt_btn.click(
        save_and_refresh_prompts,
        inputs=[new_prompt, pprompt, nprompt],
        outputs=[saved_prompts]
    )
    del_prompt_btn.click(
        delete_and_refresh_prompts,
        inputs=[saved_prompts],
        outputs=[saved_prompts]
    )
    reload_prompts_btn.click(
        refresh_prompt_list,
        inputs=[],
        outputs=[saved_prompts]
    )
    load_prompt_btn.click(
        config.get_prompt,
        inputs=[saved_prompts],
        outputs=[pprompt, nprompt]
    )

    # Return the dictionary with all UI components
    return {
        'saved_prompts': saved_prompts,
        'in_pprompt': pprompt,
        'in_nprompt': nprompt
    }

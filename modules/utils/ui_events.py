"""sd.cpp-webui - UI interactivity and events module"""

import gradio as gr

import modules.utils.queue as queue_manager
from modules.shared_instance import (
    sd_options, model_state
)


def get_ordered_inputs(inputs_map):
    """Utility to ensure keys and components always match up."""
    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[k] for k in ordered_keys]
    return ordered_keys, ordered_components


def bind_generation_pipeline(api_func, ordered_keys, ordered_components, outputs_map):
    """Connects the UI components to the generation queue."""

    def submit_job(*args):
        params = dict(zip(ordered_keys, args))

        queue_manager.add_job(api_func, params)

        q_len = queue_manager.get_queue_size()

        print(f"\n\nJob submitted! Position in queue: {q_len}.\n"),

        return (
            gr.update(visible=True, value=0),
            gr.update(visible=True, value="Added to queue...")
        )

    def poll_status():
        state = queue_manager.get_status()
        q_len = queue_manager.get_queue_size()

        if not state["is_running"] and q_len == 0:
            if state.get("is_finished"):
                state["is_finished"] = False
            else:
                return (
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

        prog = state["progress"]
        stat = state["status"]

        if not state["is_running"] and q_len > 0:
            prog = gr.skip()
            stat = gr.skip()
        elif prog == 0:
            prog = gr.skip()
            stat = gr.skip()

        queue_display = gr.update(
            value=f"⏳ Jobs in queue: {q_len}" if q_len > 0 else "",
            visible=(q_len > 0)
        )

        return (
            state["command"],
            prog,
            stat,
            state["stats"],
            state["images"],
            gr.skip(),
            queue_display
        )

    outputs_map['gen_btn'].click(
        fn=submit_job,
        inputs=ordered_components,
        outputs=[
            outputs_map['progress_slider'],
            outputs_map['progress_textbox'],
        ]
    )

    outputs_map['timer'].tick(
        poll_status,
        inputs=[],
        outputs=[
            outputs_map['command'],
            outputs_map['progress_slider'],
            outputs_map['progress_textbox'],
            outputs_map['stats'],
            outputs_map['img_final'],
            outputs_map['timer'],
            outputs_map['queue_tracker']
        ]
    )


def apply_lora(
    lora_model, lora_strength, lora_prompt_switch,
    pprompt, nprompt
):
    if lora_model:
        lora_string = "<lora:" + lora_model + ":" + str(lora_strength) + ">"
        n_lora_string = "<lora:" + lora_model + ":-" + str(lora_strength) + ">"

        if lora_prompt_switch == "Positive":
            pprompt = "".join([pprompt, lora_string])

        elif lora_prompt_switch == "Negative":
            if nprompt is True:
                nprompt = "".join([nprompt, lora_string])
            elif nprompt.visible is False:
                pprompt = "".join([pprompt, n_lora_string])

    return (pprompt, nprompt)


def unet_tab_switch(*args):
    """Switches to the UNET tab"""
    return (
        gr.update(value=1),                                                # + UNET Tab
        gr.update(value=None),                                             # - Checkpoint Model
        gr.update(value=model_state.bak_unet_model),                       # + UNET Model
        gr.update(value=None),                                             # - Checkpoint VAE
        gr.update(value=model_state.bak_unet_vae),                         # + UNET VAE
        gr.update(value=model_state.bak_clip_g),                           # + clip_g
        gr.update(value=model_state.bak_clip_l),                           # + clip_l
        gr.update(value=model_state.bak_t5xxl),                            # + t5xxl
        gr.update(value=model_state.bak_llm),                              # + llm
        gr.update(value=model_state.bak_guidance_bool, visible=True),      # + guidance_bool
        gr.update(visible=True),                                           # + guidance
        gr.update(value=model_state.bak_flow_shift_bool, visible=True),    # + flow_shift_bool
        gr.update(visible=True),                                           # + flow_shift
    )


def ckpt_tab_switch(*args):
    """Switches to the checkpoint tab"""
    return (
        gr.update(value=0),                             # + Checkpoint Tab
        gr.update(value=model_state.bak_ckpt_model),    # + Checkpoint Model
        gr.update(value=None),                          # - UNET Model
        gr.update(value=model_state.bak_ckpt_vae),      # + Checkpoint VAE
        gr.update(value=None),                          # - UNET VAE
        gr.update(value=None),                          # - clip_g
        gr.update(value=None),                          # - clip_l
        gr.update(value=None),                          # - t5xxl
        gr.update(value=None),                          # - llm
        gr.update(value=False, visible=False),          # - guidance_bool
        gr.update(visible=False),                       # - guidance
        gr.update(value=False, visible=False),          # - flow_shift_bool
        gr.update(visible=False),                       # - flow_shift
    )


def update_interactivity(count, checkbox_value):
    """
    Generates a specified number of gr.update objects to set interactivity.
    """
    is_interactive = bool(checkbox_value)

    if count == 1:
        return gr.update(interactive=is_interactive)

    return tuple(gr.update(interactive=is_interactive) for _ in range(count))


def refresh_all_options():
    sd_options.refresh()
    return [
        gr.update(choices=sd_options.get_opt("samplers")),
        gr.update(choices=sd_options.get_opt("schedulers")),
        gr.update(choices=["none"] + sd_options.get_opt("previews")),
        gr.update(choices=["Default"] + sd_options.get_opt("prediction"))
    ]

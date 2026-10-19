"""sd.cpp-webui - UI interactivity and events module"""

import itertools
import os
import threading

import gradio as gr

import modules.utils.streaming_queue as queue_manager
from modules.shared_instance import (
    sd_options, model_state
)
from modules.utils.sdcpp_utils import (
    get_cached_devices, refresh_device_cache
)

# Populate the device cache on the first launch (mirrors sd_options)
if not get_cached_devices():
    refresh_device_cache()


_polling_configs = []


def get_ordered_inputs(inputs_map):
    """Utility to ensure keys and components always match up."""
    ordered_keys = sorted(inputs_map.keys())
    ordered_components = [inputs_map[k] for k in ordered_keys]
    return ordered_keys, ordered_components


def bind_generation_pipeline(
    api_func, ordered_keys, ordered_components, outputs_map
):
    """
    Connects the UI components to the generation queue.
    """
    tab_id = api_func.__name__

    _pending_jobs = {}
    _pending_lock = threading.Lock()
    _stream_lock = threading.Lock()
    _job_counter = itertools.count(1)

    trigger = gr.Textbox(
        value="",
        visible=False,
        interactive=False,
        elem_id=f"gen-trigger-{tab_id}"
    )

    def enqueue_job(*args):
        params = dict(zip(ordered_keys, args))
        sub = queue_manager.add_job(api_func, params, owner=tab_id)
        job_id = str(next(_job_counter))
        with _pending_lock:
            _pending_jobs[job_id] = sub
        return job_id

    def _drain_job(sub):
        last_imgs = gr.skip()
        while True:
            kind, payload = sub.get()

            if kind == "done":
                remaining = queue_manager.get_queue_size()
                if remaining > 0:
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        last_imgs,
                        gr.update(
                            visible=True,
                            value=f"⏳ Jobs in queue: {remaining}"
                        ),
                    )
                else:
                    yield (
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        last_imgs,
                        gr.update(visible=False, value=""),
                    )
                return

            if kind == "stopped":
                yield (
                    gr.update(value=""),
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(value=""),
                    gr.update(value=None),
                    gr.update(visible=False, value=""),
                )
                return

            if kind == "error":
                yield (
                    gr.skip(),
                    gr.update(visible=False, value=0),
                    gr.update(visible=False, value=""),
                    gr.update(value=f"Error: {payload}"),
                    last_imgs,
                    gr.update(visible=False, value=""),
                )
                return

            cmd, prog, prog_txt, stats, imgs = payload

            if imgs is not None:
                last_imgs = imgs

            remaining = queue_manager.get_queue_size()
            if remaining > 0:
                queue_display = gr.update(
                    visible=True,
                    value=f"⏳ Jobs in queue: {remaining}",
                )
            else:
                queue_display = gr.update(visible=False, value="")

            yield (cmd, prog, prog_txt, stats, imgs, queue_display)

    def stream_jobs(_trigger_value):
        with _stream_lock:
            did_work = False
            while True:
                with _pending_lock:
                    if not _pending_jobs:
                        if not did_work:
                            yield (
                                gr.skip(),
                                gr.skip(),
                                gr.skip(),
                                gr.skip(),
                                gr.skip(),
                                gr.skip(),
                            )
                        return
                    job_id = next(iter(_pending_jobs))
                    sub = _pending_jobs.pop(job_id)

                did_work = True

                yield (
                    gr.skip(),
                    gr.update(visible=True, value=0),
                    gr.update(visible=True, value="Starting..."),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                )

                try:
                    yield from _drain_job(sub)
                except Exception as exc:
                    yield (
                        gr.skip(),
                        gr.update(visible=False, value=0),
                        gr.update(visible=False, value=""),
                        gr.update(
                            value=f"Error: {type(exc).__name__}: {exc}"
                        ),
                        None,
                        gr.update(visible=False, value=""),
                    )

    outputs_map['gen_btn'].click(
        fn=enqueue_job,
        inputs=ordered_components,
        outputs=[trigger],
    )

    trigger.change(
        fn=stream_jobs,
        inputs=[trigger],
        outputs=[
            outputs_map['command'],
            outputs_map['progress_slider'],
            outputs_map['progress_textbox'],
            outputs_map['stats'],
            outputs_map['img_final'],
            outputs_map['queue_tracker'],
        ],
        concurrency_limit=None,
    )


def apply_lora(
    lora_model, lora_strength, lora_prompt_switch,
    pprompt, nprompt
):
    if lora_model:
        lora_model = os.path.splitext(lora_model)[0]
        lora_string = "<lora:" + lora_model + ":" + str(lora_strength) + ">"
        n_lora_string = "<lora:" + lora_model + ":-" + str(lora_strength) + ">"

        if lora_prompt_switch == "Positive":
            pprompt = "".join([pprompt or "", lora_string])

        elif lora_prompt_switch == "Negative":
            nprompt = "".join([nprompt or "", n_lora_string])

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
        gr.update(value=model_state.bak_llm_vision),                       # + llm_vision
        gr.update(value=model_state.bak_tokenizer),                        # + tokenizer
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
        gr.update(value=None),                          # - llm_vision
        gr.update(value=None),                          # - tokenizer
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
    refresh_device_cache()
    return [
        gr.update(choices=sd_options.get_opt("samplers")),
        gr.update(choices=sd_options.get_opt("schedulers")),
        gr.update(choices=sd_options.get_opt("previews")),
        gr.update(choices=["Default"] + sd_options.get_opt("prediction"))
    ]

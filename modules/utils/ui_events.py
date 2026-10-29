"""sd.cpp-webui - UI interactivity and events module"""

import itertools
import os
import threading

import gradio as gr

import modules.utils.streaming_queue as queue_manager
from modules.ui.progress import progress_update
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


# Output slots of the generation pipeline, in order
_VIEW_SLOTS = (
    'command', 'progress_bar', 'stats', 'img_final', 'queue_tracker'
)

# Class of the output galleries
OUTPUT_GALLERY_CLASS = "gen-output"

# Output galleries are square: their height follows their default width,
# and the images fit their cell, whatever their aspect ratio (Gradio makes
# every item a full-width square, taller than the gallery on most screens).
# The selectors outweigh Gradio's scoped ones.
# Output images can't be clicked while generating.
GALLERY_CSS = """
.gen-output.gen-output .grid-wrap {
    overflow: hidden;
}
.gen-output.gen-output .grid-wrap.fixed-height {
    height: auto;
    aspect-ratio: 1 / 1;
}
.gen-output.gen-output .grid-container {
    height: 100%;
    grid-template-rows: repeat(var(--grid-rows), minmax(0, 1fr));
    grid-auto-rows: minmax(0, 1fr);
}
.gen-output.gen-output .gallery-item {
    min-height: 0;
    min-width: 0;
}
.gen-output.gen-output .thumbnail-item {
    aspect-ratio: auto;
    height: 100%;
    width: 100%;
}
.gen-output.gen-output .thumbnail-item img {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
.gen-busy .thumbnail-item {
    pointer-events: none;
}
"""

# Clicks every tab's resync button when the page becomes visible again,
# so a stream dropped while the page was hidden is restored
RESYNC_JS = """
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState !== 'visible') return;
        document.querySelectorAll('.gen-resync').forEach((el) => {
            const btn = el.tagName === 'BUTTON' ? el : el.querySelector('button');
            if (btn) btn.click();
        });
    });
"""


def _comparable(value) -> bool:
    """True for values cheap to compare (not images)."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_comparable(v) for v in value)
    if isinstance(value, dict):
        return all(_comparable(v) for v in value.values())
    return False


def _same(a: dict, b: dict) -> bool:
    """Whether two slot states are known to be equal."""
    return _comparable(a) and _comparable(b) and a == b


def _merge_update(current: dict, update) -> dict:
    """
    Merges a component update into its current state.
    gr.skip() leaves it unchanged, a plain value replaces it.
    """
    if isinstance(update, dict) and update.get('__type__') == 'update':
        merged = dict(current)
        merged.update(
            {k: v for k, v in update.items() if k != '__type__'}
        )
        return merged
    return {'value': update}


class GenerationStream:
    """
    Collects the job updates of a tab server-side, independently of the
    browser, and streams them. A stream dropped while the page was
    hidden loses nothing: a new stream sends the current state and
    resumes streaming.
    """

    def __init__(self, api_func):
        self.api_func = api_func
        self.tab_id = api_func.__name__

        # Latest state of each output slot
        self.view = {slot: {} for slot in _VIEW_SLOTS}
        # Version of the last change of each slot
        self.slot_versions = {slot: 0 for slot in _VIEW_SLOTS}
        self.cond = threading.Condition()
        self.version = 0
        self.active = 0
        # Newest stream of each browser session
        self.streams = {}
        self.job_counter = itertools.count(1)

    def _set_view(self, *updates):
        """Applies one update per output slot and wakes the streams."""
        with self.cond:
            changed = False
            for slot, update in zip(_VIEW_SLOTS, updates):
                if update == gr.skip():
                    continue
                merged = _merge_update(self.view[slot], update)
                # The CLI preview keeps its path while its content changes,
                # so images are always sent again
                if slot != 'img_final' and _same(merged, self.view[slot]):
                    continue
                self.view[slot] = merged
                self.slot_versions[slot] = self.version + 1
                changed = True
            if changed:
                self.version += 1
                self.cond.notify_all()

    @staticmethod
    def _queue_display():
        remaining = queue_manager.get_queue_size()
        if remaining > 0:
            return gr.update(
                visible=True,
                value=f"⏳ Jobs in queue: {remaining}"
            )
        return gr.update(visible=False, value="")

    def _consume_job(self, sub):
        """Collects the updates of a job, with or without a browser."""
        last_imgs = gr.skip()
        try:
            while True:
                kind, payload = sub.get()

                if kind == "done":
                    self._set_view(
                        gr.skip(),
                        gr.skip(),
                        gr.skip(),
                        last_imgs,
                        self._queue_display(),
                    )
                    return

                if kind == "stopped":
                    # Keep the command, clear the rest
                    self._set_view(
                        gr.skip(),
                        progress_update(visible=False),
                        gr.update(visible=False, value=""),
                        gr.update(value=None),
                        gr.update(visible=False, value=""),
                    )
                    return

                if kind == "error":
                    self._set_view(
                        gr.skip(),
                        progress_update(visible=False),
                        gr.update(visible=True, value=f"Error: {payload}"),
                        last_imgs,
                        gr.update(visible=False, value=""),
                    )
                    return

                cmd, prog, stats, imgs = payload

                if imgs is not None:
                    last_imgs = imgs

                self._set_view(cmd, prog, stats, imgs, self._queue_display())
        finally:
            with self.cond:
                self.active -= 1
                self.version += 1
                self.cond.notify_all()

    def enqueue(self, params: dict) -> str:
        """Queues a job. Returns a new trigger value."""
        with self.cond:
            idle = self.active == 0
            self.active += 1
        if idle:
            # Statistics are only shown at the end of a generation
            self._set_view(
                gr.skip(),
                progress_update(0, status="Starting..."),
                gr.update(visible=False, value=""),
                gr.skip(),
                gr.skip(),
            )
        sub = queue_manager.add_job(self.api_func, params, owner=self.tab_id)
        threading.Thread(
            target=self._consume_job, args=(sub,), daemon=True
        ).start()
        return str(next(self.job_counter))

    def _snapshot(self, sent: dict) -> tuple:
        """
        Builds the updates of the slots changed since `sent` (the slot
        versions a stream already delivered), the others are skipped.
        """
        snapshot = []
        idle = self.active == 0
        for slot in _VIEW_SLOTS:
            changed = self.slot_versions[slot] != sent.get(slot)
            update = dict(self.view[slot]) if changed else {}
            if slot == 'img_final' and (changed or idle != sent.get('idle')):
                # The images can't be clicked or enlarged while generating
                update.update(
                    allow_preview=idle,
                    elem_classes=(
                        [OUTPUT_GALLERY_CLASS] if idle
                        else [OUTPUT_GALLERY_CLASS, 'gen-busy']
                    ),
                )
            snapshot.append(gr.update(**update) if update else gr.skip())
            sent[slot] = self.slot_versions[slot]
        sent['idle'] = idle
        return tuple(snapshot)

    def stream(self, session=None):
        """
        Streams the collected state until the jobs of this tab end, each
        output only when it changes. Only the newest stream of a session
        runs, an older one (e.g. a stream the browser dropped) stops.
        """
        with self.cond:
            stream_id = self.streams.get(session, 0) + 1
            self.streams[session] = stream_id
        delivered = -1
        # Slot versions delivered by this stream, a new one sends everything
        sent = {}

        while True:
            with self.cond:
                while (self.version == delivered
                       and self.active > 0
                       and self.streams.get(session) == stream_id):
                    self.cond.wait(timeout=1.0)
                if self.streams.get(session) != stream_id:
                    return
                delivered = self.version
                finished = self.active == 0
                snapshot = self._snapshot(sent)

            yield snapshot

            if finished:
                return


def bind_generation_pipeline(
    api_func, ordered_keys, ordered_components, outputs_map
):
    """
    Connects the UI components to the generation queue.
    """
    tab_id = api_func.__name__
    gen_stream = GenerationStream(api_func)

    trigger = gr.Textbox(
        value="",
        visible=False,
        interactive=False,
        elem_id=f"gen-trigger-{tab_id}"
    )
    # Clicked by RESYNC_JS when the page becomes visible again
    resync_btn = gr.Button(
        value="Resync",
        visible="hidden",
        elem_id=f"gen-resync-{tab_id}",
        elem_classes=["gen-resync"]
    )

    def enqueue_job(*args):
        return gen_stream.enqueue(dict(zip(ordered_keys, args)))

    def stream_jobs(_trigger_value=None, request: gr.Request = None):
        yield from gen_stream.stream(getattr(request, 'session_hash', None))

    stream_outputs = [outputs_map[slot] for slot in _VIEW_SLOTS]

    outputs_map['gen_btn'].click(
        fn=enqueue_job,
        inputs=ordered_components,
        outputs=[trigger],
    )

    trigger.change(
        fn=stream_jobs,
        inputs=[trigger],
        outputs=stream_outputs,
        concurrency_limit=None,
        # Gradio's generating outline only around the images
        show_progress_on=outputs_map['img_final'],
    )

    resync_btn.click(
        fn=stream_jobs,
        inputs=[],
        outputs=stream_outputs,
        concurrency_limit=None,
        show_progress="hidden",
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

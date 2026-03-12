"""sd.cpp-webui - UI components for the cache widget"""

from functools import partial

import gradio as gr

from modules.shared_instance import config
from modules.utils.ui_events import update_interactivity
from .constants import (
    CACHE_MODE, CACHE_DIT_PRESET, SCM_POLICY
)


def create_cache_ui():
    """Create cache specific UI"""
    with gr.Accordion(
        label="Cache", open=False
    ):
        cache_bool = gr.Checkbox(
            label="Enable cache",
            value=config.get('def_cache_bool')
        )

        cache_mode = gr.Dropdown(
            label="Cache mode",
            choices=CACHE_MODE,
            value=config.get('def_cache_mode'),
            interactive=False
        )

        cache_dit_preset = gr.Dropdown(
            label="cache-dit preset",
            choices=CACHE_DIT_PRESET,
            value=config.get('def_cache_dit_preset'),
            interactive=False
        )

        with gr.Accordion(
            label="Advanced", open=False
        ):
            cache_option = gr.Textbox(
                label="Cache option",
                placeholder=(
                    "named cache params (key=value format, comma-separated):\n"
                    "- easycache/ucache:\n"
                    "threshold=,start=,end=,decay=,relative=,reset=\n"
                    "- dbcache/taylorseer/cache-dit:\n"
                    "Fn=,Bn=,threshold=,warmup=\n"
                    "Examples: \"threshold=0.25\" or\n"
                    '"threshold=1.5,reset=0"'
                ),
                lines=7,
            )

            scm_mask = gr.Textbox(
                label="SCM Mask",
                placeholder=(
                    "SCM steps mask for cache-dit:\n"
                    "comma-separated 0/1 (e.g., \"1,1,1,0,0,1,0,0,1,0\") - 1=compute, 0=can cache"
                ),
                lines=2,
            )

            scm_policy = gr.Dropdown(
                label="SCM Policy",
                choices=SCM_POLICY,
                value=config.get('def_scm_policy')
            )

    cache_comp = [
        cache_mode, cache_dit_preset, cache_option,
        scm_mask, scm_policy
    ]

    cache_bool.change(
        partial(update_interactivity, len(cache_comp)),
        inputs=cache_bool,
        outputs=cache_comp
    )

    return {
        'in_cache_bool': cache_bool,
        'in_cache_mode': cache_mode,
        'in_cache_dit_preset': cache_dit_preset,
        'in_cache_option': cache_option,
        'in_scm_mask': scm_mask,
        'in_scm_policy': scm_policy
    }

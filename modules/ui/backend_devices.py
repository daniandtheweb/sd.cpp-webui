"""sd.cpp-webui - UI component for backend device assignment"""

import math

import gradio as gr

from modules.shared_instance import config
from modules.utils.sdcpp_utils import (
    build_device_choices, get_cached_devices
)

# Modules the CLI knows about (see SDBackendModule in the sd.cpp source)
MODULES = [
    "diffusion",
    "te",
    "clip_vision",
    "vae",
    "controlnet",
    "photomaker",
    "upscaler",
    "detector",
    "audio_encoder",
]
# Only these modules support layer/row splitting of their compute
SPLIT_MODULES = ("diffusion", "te")
SPLIT_MODES = ["default", "layer", "row"]
AUTO_FIT_MODES = ["on", "off"]


def _budget_value(raw):
    """Coerces a saved budget entry to a finite float (None if invalid)."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tokenize_devices(device):
    """
    Splits a stored device value into its device tokens.
    Example: 'Vulkan0&Vulkan1' -> ['Vulkan0', 'Vulkan1'], default -> [].
    """
    if isinstance(device, (list, tuple)):
        return [str(d).strip() for d in device if str(d).strip()]
    if not device or str(device).strip().lower() in ('default', 'auto', ''):
        return []
    return [t.strip() for t in str(device).split('&') if t.strip()]


def _norm_device(value):
    """Normalizes a single-device selector value; '' means default."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ''
    value = str(value or '').strip()
    if value.lower() in ('', 'default', 'auto'):
        return ''
    return value


def _rows_to_map(rows):
    """Converts [[component, device], ...] into a component->device map."""
    result = {}
    for row in (rows or []):
        if row and len(row) >= 2 and str(row[0]).strip():
            device = str(row[1]).strip() or 'default'
            result[str(row[0]).strip()] = device
    return result


def _sync_backend_table(state_rows, default_value, *module_values):
    """
    Rebuilds a backend table from its selectors. The default row is
    stored under '*'; device lists are joined with '&'.
    """
    rows = _rows_to_map(state_rows)
    device = _norm_device(default_value)
    rows.pop('primary', None)  # legacy key, superseded by '*'
    if device:
        rows['*'] = device
    else:
        rows.pop('*', None)

    for name, value in zip(MODULES, module_values):
        device = '&'.join(_tokenize_devices(value))
        if device:
            rows[name] = device
        else:
            rows.pop(name, None)

    # Canonical order: '*' first, module order, custom rows last
    ordered = {'*': rows['*']} if '*' in rows else {}
    for name in MODULES:
        if name in rows:
            ordered[name] = rows[name]
    for name, device in rows.items():
        if name not in ordered:
            ordered[name] = device
    return [[name, device] for name, device in ordered.items()]


def _sync_split_modes(diffusion_mode, te_mode):
    """Rebuilds the '--split-mode' assignments from the split selectors."""
    rows = []
    for name, mode in (('diffusion', diffusion_mode), ('te', te_mode)):
        mode = str(mode or 'default').strip()
        if mode in ('layer', 'row'):
            rows.append([name, mode])
    return rows


def _create_backend_accordion(label, config_key, device_choices,
                              params=False, split_initial=None):
    """
    Creates a nested backend accordion plus a gr.State carrying the
    table rows consumed by the command builders.
    Returns (state, selectors, split_selectors).
    """
    initial = _rows_to_map(config.get(config_key))
    split_initial = split_initial or {}
    extras = sorted({
        token
        for device in initial.values()
        for token in _tokenize_devices(device)
        if token not in device_choices
    })
    devices = device_choices + extras
    if params:
        default_choices = ["default", "disk", "gpu"] + devices
        single_choices = ["default", "disk"] + devices
    else:
        default_choices = ["default", "gpu"] + devices
        single_choices = ["default"] + devices

    def _initial_default():
        value = initial.get('*') or initial.get('primary') or 'default'
        if value.lower() in ('', 'default', 'auto'):
            value = 'default'
        return value

    default_value = _initial_default()
    default_choices_local = (default_choices
                             if default_value in default_choices
                             else default_choices + [default_value])

    table_selectors = []
    split_selectors = []

    def _add_module(name):
        if params or name not in SPLIT_MODULES:
            value = initial.get(name, 'default') or 'default'
            if value.lower() in ('', 'auto'):
                value = 'default'
            # Legacy multi-device values keep their first device
            if '&' in value:
                value = value.split('&')[0]
            choices = (single_choices
                       if value in single_choices
                       else single_choices + [value])
            table_selectors.append(
                gr.Dropdown(label=name, choices=choices, value=value)
            )
            return
        with gr.Row():
            multi = gr.Dropdown(
                label=name,
                choices=devices,
                multiselect=True,
                value=_tokenize_devices(initial.get(name)),
                scale=3
            )
            split = gr.Dropdown(
                label="split mode",
                choices=SPLIT_MODES,
                value=split_initial.get(name, "default"),
                scale=1
            )
            split_selectors.append(split)
        table_selectors.append(multi)

    with gr.Accordion(label=label, open=False):
        table_selectors.append(gr.Dropdown(
            label="default (all modules)",
            choices=default_choices_local,
            value=default_value
        ))
        gr.Markdown("---")
        with gr.Row():
            with gr.Column():
                for name in MODULES[:5]:
                    _add_module(name)
            with gr.Column():
                for name in MODULES[5:]:
                    _add_module(name)

    state = gr.State(value=[list(item) for item in initial.items()])
    for selector in table_selectors:
        selector.change(
            _sync_backend_table,
            inputs=[state] + table_selectors,
            outputs=[state]
        )
    return state, table_selectors, split_selectors


def create_max_vram_accordion():
    """
    Creates a nested 'Max VRAM budget (--max-vram)' accordion: one budget
    per cached device (cpu/disk excluded) plus the 'All devices' budget.
    Returns the gr.State carrying the [device, budget] rows.
    """
    devices = [
        d for d in get_cached_devices()
        if d.lower() not in ('default', 'disk', 'cpu')
    ]
    saved = {}
    for row in (config.get('def_max_vram_table') or []):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        key = str(row[0]).strip().lower()
        if key in ('', 'default', 'all'):
            key = '*'
        saved[key] = _budget_value(row[1])

    state = gr.State(value=config.get('def_max_vram_table'))
    with gr.Accordion(label="Max VRAM budget (--max-vram)", open=False):
        default_budget = gr.Number(
            label="All devices (GiB)",
            value=saved.get('*'),
            precision=1,
            minimum=0
        )
        budget_inputs = []
        for i in range(0, len(devices), 2):
            with gr.Row():
                for device in devices[i:i + 2]:
                    budget_inputs.append((device, gr.Number(
                        label=f"{device} (GiB)",
                        value=saved.get(device.lower()),
                        precision=1
                    )))

        def _sync(default_value, *values):
            rows = []
            if default_value is not None:
                rows.append(['*', str(default_value)])
            for device, value in zip(
                [d for d, _ in budget_inputs], values
            ):
                if value is not None:
                    rows.append([device, str(value)])
            return rows

        inputs = [default_budget] + [comp for _, comp in budget_inputs]
        for comp in inputs:
            comp.change(_sync, inputs=inputs, outputs=[state])
    return state


def create_backend_devices_ui():
    """
    Creates a top-level 'Backend devices' accordion holding the max VRAM
    budgets, the placement options and the runtime and parameter backends.
    Returns (backend_state, params_state, split_state, offload_to_cpu,
    max_vram_state, auto_fit).
    """
    device_choices = [d for d in build_device_choices() if d != 'default']
    split_initial = _rows_to_map(config.get('def_split_modes'))

    with gr.Accordion(label="Backend devices", open=False):
        max_vram_state = create_max_vram_accordion()
        with gr.Row():
            auto_fit = gr.Dropdown(
                label="Auto-fit (--auto-fit)",
                choices=AUTO_FIT_MODES,
                value=config.get('def_auto_fit')
            )
        with gr.Row():
            offload_to_cpu = gr.Checkbox(
                label="Offload to CPU (--offload-to-cpu)",
                value=config.get('def_offload_to_cpu')
            )
        (backend_state, backend_selectors,
         split_selectors) = _create_backend_accordion(
            "Backend Configuration (--backend)",
            'def_backend_table', device_choices,
            split_initial=split_initial
        )
        params_state, params_selectors, _ = _create_backend_accordion(
            "Parameters Backend (--params-backend)",
            'def_params_backend_table', device_choices,
            params=True
        )

        split_state = gr.State(
            value=[list(item) for item in split_initial.items()]
        )
        for selector in split_selectors:
            selector.change(
                _sync_split_modes,
                inputs=split_selectors,
                outputs=[split_state]
            )

    return (backend_state, params_state, split_state,
            offload_to_cpu, max_vram_state, auto_fit)

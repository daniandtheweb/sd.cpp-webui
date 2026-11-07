"""sd.cpp-webui - extensions - the Gradio side

Everything an extension puts on screen is rendered here: its own tabs
(before Settings), its sections inside the generation tabs, and the
Extensions tab that lists what is installed (after Settings).
"""

import html

import gradio as gr

from modules.extensions.host import (
    LOADED,
    downloads_log,
    extensions,
    registered_settings,
    registered_tabs,
    registered_uis,
    report_error,
    set_setting_value,
    setting_value,
)
from modules.extensions.loader import load_extensions

INFO_CSS = """
.ext-table { width: 100%; border-collapse: collapse; }
.ext-table th, .ext-table td {
    text-align: left; vertical-align: top; padding: 8px 10px;
    border-bottom: 1px solid var(--border-color-primary);
}
.ext-table .ext-sub { opacity: 0.7; font-size: 0.9em; }
.ext-table .ext-bad { color: var(--error-text-color, #c0392b); }
"""


def ensure_loaded():
    """
    The tab modules build their UI when they are imported, which is
    before main() runs, so the first one to need the extensions loads
    them (load_extensions only works once).
    """
    from modules.shared_instance import config
    load_extensions(config)


def render_extension_tabs():
    """
    Render the tabs extensions added, each in its own TabItem.
    Must be called inside a gr.Tabs block, before the Settings tab.
    """
    ensure_loaded()
    seen = set()
    for title, render in registered_tabs():
        if title in seen:
            print(f"Skipping duplicate tab: {title!r}")
            continue
        seen.add(title)
        try:
            with gr.TabItem(title):
                render()
        except (Exception, SystemExit):
            report_error(f"Failed to render tab {title!r}")


def create_extensions_ui(tab: str) -> dict:
    """
    Render the 'Extensions' tab of a generation tab: one accordion per
    extension that added a UI to it. Returns {param key: component} for
    the inputs map, so the values reach the hooks; empty (and nothing
    rendered) when no extension added a UI to the tab.
    """
    ensure_loaded()
    uis = registered_uis(tab)
    if not uis:
        return {}

    inputs = {}
    with gr.Tab("Extensions"):
        for ext, render in uis:
            try:
                with gr.Accordion(ext.name, open=False):
                    values = render(tab)
            except (Exception, SystemExit):
                report_error(f"{ext.id}: failed to render its {tab} UI")
                continue
            for name, component in (values or {}).items():
                if not isinstance(component, gr.components.Component):
                    print(
                        f"{ext.id}: {name!r} is not a Gradio component; "
                        "ignored"
                    )
                    continue
                inputs[ext.key(name)] = component
    return inputs


def _link(url: str) -> str:
    if not url.lower().startswith(('http://', 'https://')):
        return html.escape(url)
    safe = html.escape(url, quote=True)
    return (
        f'<a href="{safe}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(url)}</a>'
    )


def _status_cell(ext) -> str:
    if ext.status == LOADED:
        return html.escape(ext.status)
    return (
        f'<span class="ext-bad">{html.escape(ext.status)}</span>'
        f'<div class="ext-sub">{html.escape(ext.reason)}</div>'
    )


def _with_reasons(names: list[str], reasons: dict) -> str:
    """Table cell: each name, with the reason the extension gave."""
    cells = []
    for name in names:
        why = reasons.get(name)
        cells.append(
            html.escape(name)
            + (f'<div class="ext-sub">{html.escape(why)}</div>' if why
               else '')
        )
    return '<br>'.join(cells) or '—'


def _info_rows() -> str:
    rows = []
    for ext in extensions():
        meta = ext.meta
        requires = _with_reasons(meta.requires, meta.require_reasons)
        hosts = _with_reasons(ext.hosts, meta.host_reasons)
        api = f'v{ext.api_version}' if ext.api_version is not None else '—'
        rows.append(
            '<tr>'
            f'<td><b>{html.escape(ext.name)}</b>'
            f'<div class="ext-sub">{html.escape(ext.id)}</div></td>'
            f'<td>{html.escape(meta.version) or "—"}</td>'
            f'<td>{api}</td>'
            f'<td>{html.escape(meta.author) or "—"}</td>'
            f'<td>{_link(meta.url) if meta.url else "—"}</td>'
            f'<td>{html.escape(meta.description) or "—"}</td>'
            f'<td>{requires}</td>'
            f'<td>{hosts}</td>'
            f'<td>{_status_cell(ext)}</td>'
            '</tr>'
        )
    return ''.join(rows)


def _setting_component(spec, value):
    """The component that edits a setting, by what the setting is."""
    kwargs = {'label': spec.label, 'value': value, 'info': spec.info or None}
    if spec.choices:
        return gr.Dropdown(choices=list(spec.choices), **kwargs)
    if spec.secret:
        # Never show the stored value; empty means keep it
        kwargs['value'] = ''
        kwargs['placeholder'] = (
            "(set: leave empty to keep it)" if value else "(not set)"
        )
        return gr.Textbox(type='password', **kwargs)
    kind = type(spec.default)
    if kind is bool:
        return gr.Checkbox(**kwargs)
    if kind is int:
        return gr.Number(precision=0, **kwargs)
    if kind is float:
        return gr.Number(**kwargs)
    return gr.Textbox(**kwargs)


def _render_settings(ext):
    """The settings of one extension, with a button to save them."""
    keys = list(ext.settings)
    clears = {}
    with gr.Accordion(f"{ext.name} settings", open=False):
        components = []
        for key in keys:
            spec = ext.settings[key]
            components.append(
                _setting_component(spec, setting_value(ext, key))
            )
            if spec.secret:
                clears[key] = gr.Checkbox(label=f"Clear {spec.label}")
        save = gr.Button("Save")
        status = gr.Markdown()

    secret_keys = list(clears)

    def save_settings(*values):
        values, clear = values[:len(keys)], values[len(keys):]
        cleared = {k for k, c in zip(secret_keys, clear, strict=True) if c}
        changes = {}
        for key, value in zip(keys, values, strict=True):
            if ext.settings[key].secret:
                # Empty keeps the stored value, "Clear" empties it
                if key in cleared:
                    changes[key] = ''
                elif value:
                    changes[key] = value
            else:
                changes[key] = value
        try:
            # Check them all first, so a bad one doesn't save half
            for key, value in changes.items():
                ext.settings[key].coerce(value)
            for key, value in changes.items():
                set_setting_value(ext, key, value)
        except (KeyError, ValueError) as e:
            return f"Not saved: {e}"
        return "Saved."

    save.click(
        save_settings, inputs=components + list(clears.values()),
        outputs=[status]
    )


DOWNLOAD_COLUMNS = [
    'Time', 'Extension', 'Kind', 'File', 'From', 'Size', 'Status',
    "Saved to / error",
]


def _size(count) -> str:
    """A byte count for people: '7 B', '1.5 MB'."""
    if not isinstance(count, (int, float)):
        return '—'
    for unit in ('B', 'KB', 'MB'):
        if count < 1024:
            return f"{count:.0f} {unit}" if unit == 'B' else (
                f"{count:.1f} {unit}")
        count /= 1024
    return f"{count:.1f} GB"


def _download_rows() -> list[list]:
    """The permanent log of downloads, newest first, as table rows."""
    return [
        [
            r.get('time', ''), r.get('extension', ''), r.get('kind', ''),
            r.get('file', ''), r.get('host') or r.get('url', ''),
            _size(r.get('bytes')), r.get('status', ''),
            r.get('saved_to', '') if r.get('status') == 'ok'
            else r.get('error', ''),
        ]
        for r in downloads_log()
    ]


def _render_downloads():
    """The downloads log: everything extensions downloaded, kept forever."""
    gr.Markdown(
        "### Downloads\n"
        "Everything extensions downloaded through the app, kept in "
        "`user_data/extensions/downloads.jsonl`. Failed and refused "
        "attempts are listed too."
    )
    table = gr.Dataframe(
        value=_download_rows(), headers=DOWNLOAD_COLUMNS,
        interactive=False, wrap=True,
    )
    refresh = gr.Button("Refresh the list")
    refresh.click(_download_rows, outputs=[table])


def render_extensions_info():
    """
    Render the Extensions tab: what is installed, with its version,
    author, URL, description and dependencies, and the settings the
    extensions declared. Only when there is an extension. Must be called
    inside a gr.Tabs block, after Settings.
    """
    ensure_loaded()
    if not extensions():
        return
    with gr.TabItem("Extensions", id="extensions"):
        gr.HTML(
            f'<style>{INFO_CSS}</style>'
            '<table class="ext-table"><thead><tr>'
            '<th>Extension</th><th>Version</th><th>API</th><th>Author</th>'
            '<th>URL</th><th>Description</th><th>Dependencies</th>'
            '<th>Hosts</th>'
            '<th>Status</th>'
            f'</tr></thead><tbody>{_info_rows()}</tbody></table>'
        )
        for ext in registered_settings():
            _render_settings(ext)
        _render_downloads()

"""sd.cpp-webui - UI component for the generation progress bar"""

import html
from typing import Optional

import gradio as gr


PROGRESS_CSS = """
.gen-progress-bar {
    position: relative;
    height: 2rem;
    overflow: hidden;
    border: 1px solid var(--border-color-accent, var(--color-accent));
    border-radius: var(--radius-md);
    background: var(--background-fill-secondary);
}
.gen-progress-fill {
    height: 100%;
    background: var(--color-accent, #f97316);
    transition: width 0.3s ease;
}
.gen-progress-label {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: var(--text-sm);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--body-text-color);
    text-shadow: 0 0 2px var(--background-fill-primary);
}
.gen-progress-status {
    margin-top: 0.35rem;
    text-align: center;
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    color: var(--body-text-color-subdued);
}
"""


def progress_html(
    percent: float = 0, step: Optional[int] = None,
    total: Optional[int] = None, status: str = "", unit: str = "Step"
) -> str:
    """
    Builds the progress bar markup: '<unit> step/total · percent%' inside
    the bar, the status (speed and ETA) under it.
    """
    try:
        percent = max(0, min(100, int(percent)))
    except (TypeError, ValueError):
        percent = 0

    if step is not None and total:
        label = f"{unit} {int(step)}/{int(total)} · {percent}%"
    else:
        label = f"{percent}%"

    status_html = ""
    if status:
        status_html = (
            f'<div class="gen-progress-status">{html.escape(status)}</div>'
        )

    return (
        '<div class="gen-progress">'
        '<div class="gen-progress-bar">'
        f'<div class="gen-progress-fill" style="width: {percent}%;"></div>'
        f'<span class="gen-progress-label">{label}</span>'
        '</div>'
        f'{status_html}'
        '</div>'
    )


def progress_update(
    percent: float = 0, step: Optional[int] = None,
    total: Optional[int] = None, status: str = "", visible: bool = True,
    unit: str = "Step"
):
    """Builds an update for the progress bar."""
    return gr.update(
        visible=visible,
        value=progress_html(percent, step, total, status, unit)
    )


def create_progress_bar():
    """Create the progress bar"""
    return gr.HTML(
        value=progress_html(0),
        css_template=PROGRESS_CSS,
        js_on_load=None,
        visible=False
    )

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
.gen-progress-sub {
    height: 1.4rem;
    margin-top: 0.35rem;
}
.gen-progress-sub .gen-progress-label {
    font-size: var(--text-xs);
}
.gen-progress-status {
    margin-top: 0.35rem;
    text-align: center;
    font-size: var(--text-sm);
    font-variant-numeric: tabular-nums;
    color: var(--body-text-color-subdued);
}
"""


def _percent(value) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def progress_html(
    percent: float = 0, step: Optional[int] = None,
    total: Optional[int] = None, status: str = "", unit: str = "Step",
    sub: Optional[tuple] = None
) -> str:
    """
    Builds the progress bar markup: '<unit> step/total · percent%' inside
    the bar, the status (speed and ETA) under it. sub is (percent, text):
    a second, thinner bar under the first (the tile of an upscale).
    """
    percent = _percent(percent)

    if step is not None and total:
        label = f"{unit} {int(step)}/{int(total)} · {percent}%"
    else:
        label = f"{percent}%"

    status_html = ""
    if status:
        status_html = (
            f'<div class="gen-progress-status">{html.escape(status)}</div>'
        )

    sub_html = ""
    if sub:
        sub_html = (
            '<div class="gen-progress-bar gen-progress-sub">'
            '<div class="gen-progress-fill" '
            f'style="width: {_percent(sub[0])}%;"></div>'
            f'<span class="gen-progress-label">{html.escape(str(sub[1]))}'
            '</span></div>'
        )

    return (
        '<div class="gen-progress">'
        '<div class="gen-progress-bar">'
        f'<div class="gen-progress-fill" style="width: {percent}%;"></div>'
        f'<span class="gen-progress-label">{label}</span>'
        '</div>'
        f'{sub_html}'
        f'{status_html}'
        '</div>'
    )


class ProgressUpdate(dict):
    """
    A Gradio update of the progress bar that also says what it shows, for
    whoever passes the update on (an extension reading the progress of a
    generation): percent, step, total and status.
    """
    percent: float = 0
    step: Optional[int] = None
    total: Optional[int] = None
    status: str = ""


def progress_update(
    percent: float = 0, step: Optional[int] = None,
    total: Optional[int] = None, status: str = "", visible: bool = True,
    unit: str = "Step", sub: Optional[tuple] = None
):
    """Builds an update for the progress bar."""
    update = ProgressUpdate(gr.update(
        visible=visible,
        value=progress_html(percent, step, total, status, unit, sub)
    ))
    update.percent, update.step = percent, step
    update.total, update.status = total, status
    return update


def create_progress_bar():
    """Create the progress bar"""
    return gr.HTML(
        value=progress_html(0),
        css_template=PROGRESS_CSS,
        js_on_load=None,
        visible=False
    )

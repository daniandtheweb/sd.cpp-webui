"""Progress bar markup and the steps reported by the log parser."""


def test_progress_bar_shows_steps_and_status():
    from modules.ui.progress import progress_html

    html = progress_html(25, 5, 20, "Speed: 1.50 it/s | ETA: 00:03")
    assert "width: 25%;" in html
    assert ">Step 5/20 · 25%<" in html
    assert ">Speed: 1.50 it/s | ETA: 00:03<" in html


def test_progress_bar_escapes_status():
    from modules.ui.progress import progress_html

    assert "&lt;b&gt;" in progress_html(0, status="<b>")


def test_progress_bar_without_steps():
    from modules.ui.progress import progress_html

    assert ">40%<" in progress_html(40)
    # Clamped
    assert "width: 100%;" in progress_html(250)
    assert "width: 0%;" in progress_html(None)


def test_progress_bar_unit():
    from modules.ui.progress import progress_html

    assert ">Tensor 3/10 · 30%<" in progress_html(30, 3, 10, unit="Tensor")


def test_log_parser_reports_steps():
    from modules.utils.log_parser import LogParser

    parser = LogParser()
    stats = {}
    update = parser._parse_progress_update(
        "|=====>     | 5/20 - 1.50it/s", stats
    )
    assert (update["step"], update["total"]) == (5, 20)
    assert update["percent"] == 25

    # Lines without speed fall back to the step count
    update = parser._parse_progress_update("|==>  | 3/12", stats)
    assert (update["step"], update["total"]) == (3, 12)


def test_progress_bar_layout():
    """Step count inside the bar, the status under it when given."""
    from modules.ui.progress import progress_html

    html = progress_html(60, 12, 20, "Speed: 2.00 it/s | ETA: 00:04")
    bar_end = html.index('</div>', html.index('gen-progress-label'))
    assert html.index('Step 12/20') < bar_end < html.index('ETA: 00:04')
    assert 'gen-progress-status' not in progress_html(60, 12, 20)


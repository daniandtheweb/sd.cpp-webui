import gradio as gr

from modules.utils.image_utils import (
    RATIOS, RATIO_OPTIONS, apply_ratio, switch_sizes_and_ratio,
    sync_height_from_width, sync_width_from_height
)


def test_switch_sizes_and_ratio_swaps_dimensions_and_ratio():
    assert switch_sizes_and_ratio(768, 1024, '4:3') == (1024, 768, '3:4')
    assert switch_sizes_and_ratio(1280, 720, '16:9') == (720, 1280, '9:16')
    assert switch_sizes_and_ratio(1024, 1024, '1:1') == (1024, 1024, '1:1')
    assert switch_sizes_and_ratio(768, 1024, 'Free') == (1024, 768, 'Free')


def test_ratio_options_start_with_free():
    assert RATIO_OPTIONS[0] == "Free"
    assert set(RATIO_OPTIONS[1:]) == set(RATIOS.keys())


def test_apply_ratio_free_leaves_sliders_alone():
    assert apply_ratio("Free", 1024, 768) == (gr.skip(), gr.skip())
    assert apply_ratio(None, 1024, 768) == (gr.skip(), gr.skip())
    assert apply_ratio("bogus", 1024, 768) == (gr.skip(), gr.skip())


def test_apply_ratio_snaps_to_ratio():
    # 1024 * 9 / 16 = 576, already aligned
    assert apply_ratio("16:9", 1024, 768) == (
        gr.update(value=1024), gr.update(value=576)
    )
    # 1:1 keeps the width
    assert apply_ratio("1:1", 1024, 768) == (
        gr.update(value=1024), gr.update(value=1024)
    )
    # 4:3 at 1024 is already 768 high -> unchanged
    assert apply_ratio("4:3", 1024, 768) == (
        gr.update(value=1024), gr.update(value=768)
    )
    # 1000 * 3 / 4 = 750 -> snaps to 768
    assert apply_ratio("4:3", 1000, 512) == (
        gr.update(value=1000), gr.update(value=768)
    )


def test_apply_ratio_falls_back_to_height_when_width_side_overflows():
    # 9:16 at width 4096 would need height 7255 (> 4096),
    # so the height is kept and the width is computed instead:
    # 1280 * 9 / 16 = 720 -> snaps to 704
    assert apply_ratio("9:16", 4096, 1280) == (
        gr.update(value=704), gr.update(value=1280)
    )


def test_sync_height_uses_selected_ratio():
    update, last = sync_height_from_width(1024, "16:9", None)
    assert update == gr.update(value=576)
    assert last == "h"


def test_sync_width_uses_selected_ratio():
    update, last = sync_width_from_height(576, "16:9", None)
    assert update == gr.update(value=1024)
    assert last == "w"


def test_sync_skips_when_free_unknown_or_out_of_bounds():
    assert sync_height_from_width(960, "Free", None) == (gr.skip(), gr.skip())
    assert sync_height_from_width(960, None, None) == (gr.skip(), gr.skip())
    assert sync_height_from_width(960, "bogus", None) == (gr.skip(), gr.skip())
    assert sync_height_from_width(51, "16:9", None) == (gr.skip(), gr.skip())
    assert sync_height_from_width(5000, "16:9", None) == (gr.skip(), gr.skip())
    assert sync_width_from_height(1280, "Free", None) == (gr.skip(), gr.skip())
    assert sync_width_from_height(1280, None, None) == (gr.skip(), gr.skip())
    assert sync_width_from_height(7, "16:9", None) == (gr.skip(), gr.skip())
    assert sync_width_from_height(5000, "16:9", None) == (gr.skip(), gr.skip())


def test_feedback_loop_terminates():
    """
    Simulates the event chain: user drags width, the height update
    triggers height.change, which must not update the width back.
    """
    # 1. user changes width -> height is updated, last_sync="h"
    height_update, last = sync_height_from_width(1024, "16:9", None)
    assert last == "h"

    # 2. the programmatic height change fires -> ignored and state reset
    width_update, last = sync_width_from_height(
        height_update["value"], "16:9", last
    )
    assert width_update == gr.skip()
    assert last is None

    # 3. user now drags height -> width is updated, last_sync="w"
    width_update, last = sync_width_from_height(704, "16:9", last)
    assert last == "w"

    # 4. the programmatic width change fires -> ignored and state reset
    height_update, last = sync_height_from_width(
        width_update["value"], "16:9", last
    )
    assert height_update == gr.skip()
    assert last is None

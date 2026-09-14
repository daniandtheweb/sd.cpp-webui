"""sd.cpp-webui - Utilities for image processing"""

from PIL import Image

import gradio as gr


SIZE_MIN = 64
SIZE_MAX = 4096
SIZE_STEP = 64

RATIOS = {
    '1:1': (1, 1),
    '4:3': (4, 3),
    '3:4': (3, 4),
    '16:9': (16, 9),
    '9:16': (9, 16),
    '2:3': (2, 3),
    '3:2': (3, 2)
}
RATIO_OPTIONS = ['Free'] + list(RATIOS.keys())
RATIO_INVERSE = {
    '1:1': '1:1',
    '4:3': '3:4',
    '3:4': '4:3',
    '16:9': '9:16',
    '9:16': '16:9',
    '2:3': '3:2',
    '3:2': '2:3'
}


def switch_sizes(height, width):
    """Switches width and height."""
    return (width, height)


def switch_sizes_and_ratio(height, width, ratio):
    """Switches width, height and the selected ratio."""
    return width, height, RATIO_INVERSE.get(ratio, 'Free')


def _snap_size(value):
    """Snaps a value to the size step and clamps it to the slider bounds."""
    snapped = int(round(value / SIZE_STEP)) * SIZE_STEP
    return max(SIZE_MIN, min(SIZE_MAX, snapped))


def apply_ratio(ratio_name, width, height):
    """Snaps both sliders to the selected ratio."""
    if ratio_name not in RATIOS:
        return gr.skip(), gr.skip()

    rw, rh = RATIOS[ratio_name]
    raw_height = width * rh / rw
    if SIZE_MIN <= raw_height <= SIZE_MAX:
        return gr.update(value=width), gr.update(value=_snap_size(raw_height))

    return gr.update(value=_snap_size(height * rw / rh)), gr.update(value=height)


def sync_height_from_width(width, ratio_name, last_sync):
    """
    Recomputes the height from the width using the selected ratio.
    Ignores the changes caused by the height sync
    to avoid a feedback loop between the sliders.
    """
    if last_sync == 'w':
        return gr.skip(), None
    if ratio_name not in RATIOS or not width:
        return gr.skip(), gr.skip()
    # The event is bound with preprocess=False, so partial
    # out-of-range values typed into the slider can reach us.
    if not (SIZE_MIN <= width <= SIZE_MAX):
        return gr.skip(), gr.skip()
    rw, rh = RATIOS[ratio_name]
    return gr.update(value=_snap_size(width * rh / rw)), 'h'


def sync_width_from_height(height, ratio_name, last_sync):
    """
    Recomputes the width from the height using the selected ratio.
    Ignores the changes caused by the width sync
    to avoid a feedback loop between the sliders.
    """
    if last_sync == 'h':
        return gr.skip(), None
    if ratio_name not in RATIOS or not height:
        return gr.skip(), gr.skip()
    if not (SIZE_MIN <= height <= SIZE_MAX):
        return gr.skip(), gr.skip()
    rw, rh = RATIOS[ratio_name]
    return gr.update(value=_snap_size(height * rw / rh)), 'w'


def size_extractor(image):
    """Extracts width and height from an image."""
    try:
        with Image.open(image) as img:
            return img.size
    except Exception:
        return None, None


def size_updater(img_inp):
    if not img_inp:
        return (
            gr.update(), gr.update()
        )

    if isinstance(img_inp, list):
        first_img = img_inp[0]
        if isinstance(first_img, tuple):
            img_path = first_img[0]
        elif isinstance(first_img, dict) and "name" in first_img:
            img_path = first_img["name"]
        else:
            img_path = first_img
    else:
        img_path = img_inp

    width, height = size_extractor(img_path)

    if width is None or height is None:
        return (
            gr.update(), gr.update()
        )

    return (
        gr.update(value=int(width)), gr.update(value=int(height))
    )

"""sd.cpp-webui - Utilities for image processing"""

from PIL import Image

import gradio as gr


def switch_sizes(height, width):
    """Switches width and height."""
    return (width, height)


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

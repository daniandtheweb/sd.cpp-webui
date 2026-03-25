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
    if img_inp is None:
        return (
            gr.update(), gr.update()
        )
    else:
        width, height = size_extractor(img_inp)
        return (
            gr.update(value=int(width)), gr.update(value=int(height))
        )

"""sd.cpp-webui - Utilities for image processing"""

from PIL import Image


def switch_sizes(height, width):
    """Swtiches width and height."""
    return (width, height)


def size_extractor(image):
    """Extracts width and height from an image."""
    try:
        with Image.open(image) as img:
            width, height = img.size
    except Exception:
        width, height = None, None

    return (
        width, height
    )

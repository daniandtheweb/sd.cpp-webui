"""sd.cpp-webui - Utilities for image processing"""

from PIL import Image


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

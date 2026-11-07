"""Example extension: create and reuse masks with rembg.

Uses rembg (background removal) to turn a photo into a foreground
mask, saves it to its output folder as a PNG, and lets you apply a saved
mask to any other image later: a mask is a reusable artifact, not a
one-off.

The constants below describe the extension; the loader offers to install
REQUIRES (asking first) when it is missing.
"""

import os
import re
import time
from collections.abc import Callable

from PIL import Image

NAME = 'Masker'
VERSION = '1.0'
AUTHOR = 'daniandtheweb'
URL = 'https://github.com/daniandtheweb/sd.cpp-webui'
DESCRIPTION = (
    "Creates foreground masks with rembg, saves them to their own output "
    "folder and reapplies them to other images."
)
API_VERSION = 1

# [cpu] pulls onnxruntime; use [gpu] for CUDA
REQUIRES = {
    'rembg[cpu]>=2.0.87,<3':
        "finds the foreground of a photo to make the mask",
}
# rembg downloads its model itself on first use, not through the api, so
# this is a declaration for the user to read, nothing the app enforces
HOSTS = {
    'github.com': "rembg downloads its model from its GitHub releases",
    '*.githubusercontent.com': "where GitHub stores that model file",
}

# Where masks are saved: set in register() by api.add_output(). The app
# decides the folder (outputs/extensions/masker/), the extension can't.
MASK_DIR = None

_API = None  # set in register(): the app's naming for unnamed masks

_REMOVE: "Callable[[Image.Image], Image.Image] | None" = None
_REMOVE_ERROR = None


def _remove():
    """Lazily import rembg's remove(); the import is slow (onnxruntime).

    Raises RuntimeError with a user-friendly message when rembg is
    not importable (REQUIRES should normally prevent this).
    """
    global _REMOVE, _REMOVE_ERROR
    if _REMOVE is None and _REMOVE_ERROR is None:
        try:
            from rembg import remove  # pyright: ignore[reportMissingImports]
            _REMOVE = remove
        except (Exception, SystemExit) as e:
            # SystemExit: rembg calls sys.exit(1) when no onnxruntime
            # backend is installed (i.e. plain `pip install rembg`).
            _REMOVE_ERROR = (
                "rembg cannot be loaded: "
                f"{type(e).__name__}: {e} "
                '— install a backend: python -m pip install "rembg[cpu]"'
            )
    if _REMOVE_ERROR is not None:
        raise RuntimeError(_REMOVE_ERROR)
    if _REMOVE is None:  # defensive: should not happen
        raise RuntimeError("rembg not loaded")
    return _REMOVE


def _sanitize_name(name):
    """Make a name safe to use as a filename in MASK_DIR."""
    name = re.sub(r'[^\w.-]', '-', str(name or '')).strip()
    while name.startswith('.'):
        name = name[1:]
    return name or f'mask_{time.strftime("%Y%m%d-%H%M%S")}'


def _auto_name():
    """Name an unnamed mask like the app names its outputs (1, 2, ...)."""
    if _API is None:
        return f'mask_{time.strftime("%Y%m%d-%H%M%S")}'
    return os.path.splitext(os.path.basename(_API.next_output_path()))[0]


def _mask_path(name):
    """Resolve name inside MASK_DIR; return None if it escapes it.

    Guards preview/apply against path traversal even though names
    normally come from our own list_masks() dropdown.
    """
    if not name:
        return None
    root = os.path.realpath(MASK_DIR)
    path = os.path.realpath(os.path.join(root, str(name)))
    if os.path.commonpath([root, path]) != root:
        return None
    return path


def create_mask(image, name):
    """Run rembg on image, save the alpha mask: (mask, overlay, note)."""
    if image is None:
        return None, None, "Upload an image first."
    try:
        cutout = _remove()(image)
    except Exception as e:
        return None, None, f"rembg failed: {e}"
    cutout = cutout.convert('RGBA')
    mask = cutout.getchannel('A')

    safe = _sanitize_name(name) if str(name or '').strip() else _auto_name()
    try:
        os.makedirs(MASK_DIR, exist_ok=True)
    except OSError as e:
        return None, None, f"Cannot create {MASK_DIR}: {e}"
    path = os.path.join(MASK_DIR, safe + '.png')
    n = 1
    while os.path.exists(path):
        path = os.path.join(MASK_DIR, f'{safe}-{n}.png')
        n += 1
    mask.save(path)

    # Preview: foreground over red, the classic matte-look check.
    overlay = Image.composite(
        cutout.convert('RGB'),
        Image.new('RGB', cutout.size, (255, 0, 0)),
        mask
    )
    return mask, overlay, f"Saved mask: {path}"


def list_masks():
    """Names of saved masks, sorted."""
    if not os.path.isdir(MASK_DIR):
        return []
    try:
        entries = os.listdir(MASK_DIR)
    except OSError:
        return []
    return sorted(
        name for name in entries
        if name.lower().endswith(('.png', '.jpg', '.jpeg'))
    )


def preview_mask(name):
    """Return the saved mask as an L image, or None."""
    path = _mask_path(name)
    if path is None or not os.path.isfile(path):
        return None
    return Image.open(path).convert('L')


def apply_mask(image, name):
    """Composite a saved mask onto image; returns (masked rgba, note).

    Works from the saved file alone — no dependency on the original
    image the mask was created from.
    """
    if image is None:
        return None, "Upload an image first."
    if not name:
        return None, "Pick a saved mask first."
    path = _mask_path(name)
    if path is None or not os.path.isfile(path):
        return None, f"Mask not found: {name}"
    try:
        mask = Image.open(path).convert('L').resize(
            image.size, Image.Resampling.NEAREST
        )
    except Exception as e:
        return None, f"Cannot read mask {name}: {e}"
    out = image.convert('RGBA')
    out.putalpha(mask)
    return out, f"Mask {name} applied to {image.size[0]}x{image.size[1]}."


def render_tab():
    import gradio as gr

    gr.Markdown(
        "## Masker (rembg)\n\n"
        "Creates a foreground mask with "
        "[rembg](https://github.com/danielgatis/rembg) and saves it "
        "to `outputs/extensions/masker/` (it also shows in the Gallery), "
        "where you can reuse it on other images later (or load it in the "
        "img2img Mask box).\n\n"
        "> First run downloads the u2net model (~170 MB) — "
        "be patient on the first click."
    )
    with gr.Tabs():
        with gr.TabItem("Create mask"):
            with gr.Row():
                src = gr.Image(label="Source image", type='pil')
                with gr.Column():
                    name = gr.Textbox(
                        label="Mask name",
                        placeholder="optional; empty = next number (1, 2, ...)"
                    )
                    make = gr.Button("Create mask (rembg)")
                    status = gr.Textbox(
                        label='Status', interactive=False
                    )
            with gr.Row():
                out_mask = gr.Image(
                    label="Mask (alpha)", type='pil'
                )
                out_overlay = gr.Image(
                    label="Preview (foreground on red)", type='pil'
                )
            make.click(
                create_mask,
                inputs=[src, name],
                outputs=[out_mask, out_overlay, status]
            )
        with gr.TabItem("Reuse a mask"):
            with gr.Row():
                with gr.Column():
                    picker = gr.Dropdown(
                        label="Saved masks",
                        choices=list_masks(),
                        allow_custom_value=False
                    )
                    refresh = gr.Button("Refresh list")
                    apply_src = gr.Image(
                        label="New image", type='pil'
                    )
                    apply_btn = gr.Button("Apply selected mask")
                    apply_status = gr.Textbox(
                        label='Status', interactive=False
                    )
                with gr.Column():
                    preview = gr.Image(
                        label="Selected mask", type='pil'
                    )
                    out = gr.Image(
                        label="Result (masked image)", type='pil'
                    )
            refresh.click(
                lambda: gr.Dropdown(choices=list_masks()),
                outputs=[picker]
            )
            picker.change(preview_mask, inputs=[picker], outputs=[preview])
            apply_btn.click(
                apply_mask,
                inputs=[apply_src, picker],
                outputs=[out, apply_status]
            )


def mask_selected(ctx):
    """The Gallery action: a mask of the selected image."""
    with Image.open(ctx.path) as photo:
        return create_mask(photo.convert('RGB'), '')[2]


def register(api):
    global MASK_DIR, _API
    MASK_DIR = api.add_output()
    _API = api
    api.add_tab('Masker', render_tab)
    api.add_gallery_action("Make mask", mask_selected)

"""The tiles of the upscale: cutting, redrawing and joining an image.

Only Pillow here: what redraws a tile (the generation) is given to
process() as a function, so this part works without sd.cpp.
"""

import inspect
import math
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFilter

# The multiple a side must be: sd.cpp stretches anything else to the next
# multiple of 64 (and generates that), so it is also the smallest side
ALIGN = 64
# A rectangle edge that is "everywhere": the sides of a tile on the border
# of the image never fade
FAR = 100000

LINEAR, CHESS, NO_TILES = "Linear", "Chess", "None"
NO_SEAMS, BAND, HALF, HALF_PLUS = (
    "None", "Band Pass", "Half Tile", "Half Tile + Intersections")
MODES = (LINEAR, CHESS, NO_TILES)
SEAM_MODES = (NO_SEAMS, BAND, HALF, HALF_PLUS)


def side(value: int) -> int:
    """A side the generation takes as it is: a multiple of 64."""
    return max(ALIGN, -(-int(value) // ALIGN) * ALIGN)


# ------------------------------------------------------------- the pieces

def cuts(length: int, tile: int) -> list[int]:
    """Where a length is cut into tiles of about the tile size, all alike."""
    count = max(1, math.ceil(length / max(1, tile)))
    step = length / count
    return [round(i * step) for i in range(count)] + [length]


def _cells(size: tuple, tile_w: int, tile_h: int) -> tuple:
    return cuts(size[0], tile_w), cuts(size[1], tile_h)


def tile_rects(
    size: tuple[int, int], tile_w: int, tile_h: int, mode: str = LINEAR
) -> list[tuple[int, int, int, int]]:
    """
    The tiles of an image as (left, top, right, bottom), in the order they
    are redrawn: by rows, or like a chessboard (the tiles of one colour,
    then the other, so the second ones see the first ones finished).
    """
    xs, ys = _cells(size, tile_w, tile_h)
    cells = [
        (i, j, (xs[i], ys[j], xs[i + 1], ys[j + 1]))
        for j in range(len(ys) - 1) for i in range(len(xs) - 1)
    ]
    if mode == CHESS:
        cells = ([c for c in cells if (c[0] + c[1]) % 2 == 0]
                 + [c for c in cells if (c[0] + c[1]) % 2 == 1])
    return [rect for _, _, rect in cells if rect[2] > rect[0]
            and rect[3] > rect[1]]


def seam_rects(
    size: tuple[int, int], tile_w: int, tile_h: int, mode: str, width: int
) -> list[tuple[int, int, int, int]]:
    """
    The areas over the seams of the tiles: a band along every cut (Band
    Pass, of the given width; Half Tile, as wide as a tile, so the tiles
    are shifted by half a tile), and for Half Tile + Intersections the
    squares where the cuts cross.
    """
    if mode not in (BAND, HALF, HALF_PLUS):
        return []
    xs, ys = _cells(size, tile_w, tile_h)
    wide_x, wide_y = (width, width) if mode == BAND else (tile_w, tile_h)
    rects = []
    for x in xs[1:-1]:
        for y0, y1 in zip(ys, ys[1:]):
            rects.append((x - wide_x // 2, y0, x + wide_x // 2, y1))
    for y in ys[1:-1]:
        for x0, x1 in zip(xs, xs[1:]):
            rects.append((x0, y - wide_y // 2, x1, y + wide_y // 2))
    if mode == HALF_PLUS:
        for x in xs[1:-1]:
            for y in ys[1:-1]:
                rects.append((x - tile_w // 2, y - tile_h // 2,
                              x + tile_w // 2, y + tile_h // 2))
    clamped = [(max(0, a), max(0, b), min(size[0], c), min(size[1], d))
               for a, b, c, d in rects]
    return [r for r in clamped if r[2] > r[0] and r[3] > r[1]]


class Stage(NamedTuple):
    """One go over the image: the areas to redraw and how."""
    name: str
    rects: list
    padding: int
    blur: int
    denoise: float
    dim: bool = False           # dim what is not redrawn yet (the preview)
    uniform: bool = False       # every tile is sent at the same size


def stages(size: tuple[int, int], tile_w: int, tile_h: int, *, mode: str,
           padding: int, blur: int, denoise: float, uniform: bool,
           seam: str, seam_width: int, seam_padding: int, seam_blur: int,
           seam_denoise: float) -> list[Stage]:
    """
    The stages of an upscale from its settings. The padding is at least
    twice the edge blur: the soft edge of a tile has to fade into the
    redrawn pixels of the tile next to it, not into the old ones.
    """
    result = []
    if mode in (LINEAR, CHESS):
        result.append(Stage(
            "Tiles", tile_rects(size, tile_w, tile_h, mode),
            max(padding, 2 * blur), blur, denoise, dim=True,
            uniform=uniform))
    rects = seam_rects(size, tile_w, tile_h, seam, seam_width)
    if rects:
        result.append(Stage(
            "Seams", rects, max(seam_padding, 2 * seam_blur), seam_blur,
            seam_denoise))
    return result


# ------------------------------------------------------------ redrawing

def _span(low: int, high: int, length: int, padding: int,
          target: int | None = None) -> tuple:
    """
    A range with the padding around it, grown inside the image to the side
    the generation takes (so nothing has to be stretched), or to the same
    side for every tile when target is given. Only an image shorter than
    that side leaves it short.
    """
    if target is None:
        low, high = max(0, low - padding), min(length, high + padding)
        target = side(high - low)
    else:
        middle = (low + high) // 2
        low, high = middle - target // 2, middle + (target + 1) // 2
        shift = max(0, -low) - max(0, high - length)
        low, high = low + shift, high + shift
        low, high = max(0, low), min(length, high)
    extra = min(length, target) - (high - low)
    after = min(extra, length - high)
    return low - min(extra - after, low), high + after


def _crop_box(core: tuple, size: tuple[int, int], padding: int,
              target: tuple | None = None) -> tuple:
    """The tile with its padding, as a box in the image."""
    tx, ty = target if target else (None, None)
    left, right = _span(core[0], core[2], size[0], padding, tx)
    top, bottom = _span(core[1], core[3], size[1], padding, ty)
    return left, top, right, bottom


def _mask(crop_size: tuple, core: tuple, box: tuple, size: tuple,
          padding: int, blur: int) -> Image.Image:
    """
    What of the redrawn crop is used: the tile, a little more where it
    meets another tile, with soft edges so they melt together. The sides on
    the border of the image are used up to the edge.
    """
    grow = min(2 * blur, padding)
    left = -FAR if core[0] == 0 else core[0] - box[0] - grow
    top = -FAR if core[1] == 0 else core[1] - box[1] - grow
    right = FAR if core[2] == size[0] else core[2] - box[0] + grow - 1
    bottom = FAR if core[3] == size[1] else core[3] - box[1] + grow - 1
    mask = Image.new('L', crop_size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else mask


def redraw(canvas: Image.Image, core: tuple, tile_fn, padding: int,
           blur: int, denoise: float, target: tuple | None = None):
    """
    Redraw one area of the canvas in place: tile_fn(crop, denoise) gets the
    area with its padding (a multiple of 64 on each side) and returns it
    redrawn, or is a generator that yields what it shows while it works
    (a preview image, or (image or None, progress)) and returns it. A
    generator itself: it yields (preview, box, progress) for each, with the
    box of the area in the canvas.
    """
    box = _crop_box(core, canvas.size, padding, target)
    crop = canvas.crop(box)
    sent = crop.resize((side(crop.width), side(crop.height)), Image.LANCZOS)
    made = tile_fn(sent, denoise)
    if inspect.isgenerator(made):
        while True:
            try:
                shown = next(made)
            except StopIteration as finished:
                made = finished.value
                break
            preview, progress = (
                shown if isinstance(shown, tuple) else (shown, None))
            yield preview, box, progress
    done = made.convert('RGB')
    if done.size != crop.size:
        done = done.resize(crop.size, Image.LANCZOS)
    mask = _mask(crop.size, core, box, canvas.size, padding, blur)
    canvas.paste(Image.composite(done, crop, mask), box[:2])


class Step(NamedTuple):
    """Where the work is, for the progress and the preview."""
    done: int
    total: int
    canvas: Image.Image
    current: tuple | None       # the area being redrawn (None: finished)
    remaining: list             # the areas of this stage still to do
    stage: Stage | None
    live: tuple | None = None   # (preview or None, box) of the area drawn
    progress: tuple | None = None   # what the area being drawn reports


def process(image: Image.Image, tile_fn, plan: list[Stage]):
    """
    Redraw an image stage by stage. A generator: it yields a Step before
    each area (and one at the end) and returns the new image.
    """
    canvas = image.convert('RGB')
    total = sum(len(stage.rects) for stage in plan)
    done = 0
    for stage in plan:
        target = None
        if stage.uniform and stage.rects:
            target = (side(max(r[2] - r[0] for r in stage.rects)
                           + 2 * stage.padding),
                      side(max(r[3] - r[1] for r in stage.rects)
                           + 2 * stage.padding))
        for index, core in enumerate(stage.rects):
            rest = stage.rects[index + 1:]
            yield Step(done, total, canvas, core, rest, stage)
            drawing = redraw(canvas, core, tile_fn, stage.padding,
                             stage.blur, stage.denoise, target)
            while True:
                try:
                    preview_, box, progress = next(drawing)
                except StopIteration:
                    break
                yield Step(done, total, canvas, core, rest, stage,
                           (preview_, box), progress)
            done += 1
    yield Step(done, total, canvas, None, [], None)
    return canvas


# ------------------------------------------------------------ the preview

ACCENT = (249, 115, 22)       # the orange of the app
DIM = 110                     # how dark what is not redrawn yet is (of 255)
PREVIEW_SIDE = 1024


def _scaled(core: tuple, factor: float) -> tuple:
    return tuple(round(v * factor) for v in core)


def _show_live(frame: Image.Image, step: Step, factor: float):
    """Put what the model is drawing of the current area in the frame."""
    live, box = step.live
    core = step.current
    drawn = live.convert('RGB').resize(
        (max(1, round((box[2] - box[0]) * factor)),
         max(1, round((box[3] - box[1]) * factor))), Image.BILINEAR)
    left, top, right, bottom = _scaled(core, factor)
    origin = (round(box[0] * factor), round(box[1] * factor))
    part = drawn.crop((left - origin[0], top - origin[1],
                       right - origin[0], bottom - origin[1]))
    frame.paste(part, (left, top))


def preview(step: Step, side_: int = PREVIEW_SIDE) -> Image.Image:
    """
    A frame for the user: the image as it is now, what is not redrawn yet
    dimmed (so the detail appears tile by tile), and the area being
    redrawn, only that one, lit with a glow and corner marks.
    """
    canvas = step.canvas
    factor = min(1.0, side_ / max(canvas.size))
    size = (max(1, round(canvas.width * factor)),
            max(1, round(canvas.height * factor)))
    frame = canvas.resize(size, Image.LANCZOS).convert('RGBA')
    if step.live and step.live[0] is not None:
        _show_live(frame, step, factor)
    layer = Image.new('RGBA', size, (0, 0, 0, 0))
    paint = ImageDraw.Draw(layer)

    if step.stage and step.stage.dim:
        shown = step.live is not None and step.live[0] is not None
        dimmed = step.remaining + (
            [step.current] if step.current and not shown else [])
        for core in dimmed:
            paint.rectangle(_scaled(core, factor), fill=(0, 0, 0, DIM))

    if step.current:
        left, top, right, bottom = _scaled(step.current, factor)
        paint.rectangle((left, top, right, bottom), fill=(*ACCENT, 38))
        for grow, alpha in ((6, 40), (4, 90), (2, 170), (0, 255)):
            paint.rectangle(
                (left - grow, top - grow, right + grow, bottom + grow),
                outline=(*ACCENT, alpha), width=2)
        mark = max(8, min(right - left, bottom - top) // 6)
        for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1),
                             (left, bottom, 1, -1), (right, bottom, -1, -1)):
            paint.line((x, y, x + dx * mark, y), fill=(255, 255, 255, 255),
                       width=3)
            paint.line((x, y, x, y + dy * mark), fill=(255, 255, 255, 255),
                       width=3)
    return Image.alpha_composite(frame, layer).convert('RGB')

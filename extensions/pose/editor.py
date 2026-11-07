"""The pose editor: a canvas to move the points of the skeletons.

The editor itself is JavaScript (editor.js) inside a gr.HTML. Its value is
a JSON text {w, h, photo, box, people}: the canvas size, the photo behind
it and where it is (box: two corners, in 0..1 of the canvas), and the
people. The canvas size is the user's, whatever the photo's. This module
builds it from the detected
people and reads it back, checking everything the browser sends.
"""

import base64
import io
import json
import math
from pathlib import Path

import gradio as gr
from PIL import Image

from . import skeleton

DEFAULT_SIZE = (512, 768)   # the canvas of a new editor
POINTS = len(skeleton.COLORS)
MAX_PEOPLE = 20
MAX_SIDE = 8192
PHOTO_SIDE = 640          # the photo behind the editor is shown this small

SCRIPT = Path(__file__).with_name('editor.js')

STYLE = """
.pose-bar { display: flex; gap: 8px; align-items: center;
            flex-wrap: wrap; margin-bottom: 6px; }
.pose-bar button { padding: 4px 10px; border-radius: 6px; cursor: pointer;
                   border: 1px solid var(--border-color-primary); }
.pose-hint { font-size: 0.85em; opacity: 0.7; }
.pose-wrap { display: flex; gap: 12px; flex-wrap: wrap;
             align-items: flex-start; }
.pose-stage { flex: 2 1 320px; min-width: 0; }
.pose-list { flex: 1 1 240px; min-width: 220px; display: flex;
             flex-direction: column; gap: 6px; }
.pose-list-head { display: flex; justify-content: space-between;
                  align-items: center; font-weight: 600; }
.pose-row { display: grid; grid-template-columns: 1fr auto;
            gap: 2px 8px; padding: 6px 8px; border-radius: 6px;
            cursor: pointer;
            border: 1px solid var(--border-color-primary); }
.pose-row.selected { outline: 2px solid var(--color-accent, #f97316); }
.pose-row input[type=range] { grid-column: 1; width: 100%; }
.pose-row button { grid-row: 1 / span 2; grid-column: 2; }
.pose-list button { padding: 2px 8px; border-radius: 6px; cursor: pointer;
                    border: 1px solid var(--border-color-primary); }
canvas { display: block; border-radius: 6px; cursor: crosshair; }
"""


def fit_box(canvas: tuple[int, int], photo: tuple[int, int]) -> list[float]:
    """Where a photo goes on a canvas: as big as fits, centered."""
    fit = min(canvas[0] / photo[0], canvas[1] / photo[1])
    width, height = photo[0] * fit / canvas[0], photo[1] * fit / canvas[1]
    return [(1 - width) / 2, (1 - height) / 2,
            (1 + width) / 2, (1 + height) / 2]


def dump(people: list, size: tuple[int, int],
         photo: Image.Image | None = None, box: list | None = None) -> str:
    """The editor's value: people are lists of 18 (x, y) or None."""
    if photo is not None and box is None:
        box = fit_box(size, photo.size)
    return json.dumps({
        'w': size[0], 'h': size[1],
        'photo': photo_url(photo) if photo is not None else '',
        'box': box if photo is not None else None,
        'people': [
            [None if p is None else [p[0], p[1]] for p in person]
            for person in people
        ],
    })


def blank(width: int, height: int) -> str:
    """An empty canvas of that size."""
    width, height = (_side(width), _side(height))
    return dump([], (width, height))


def add_people(value, found: list, photo: Image.Image) -> str:
    """
    The editor's value with the people found in photo added to the ones
    it has (found: lists of 18 points, 0..1 of the photo), and the photo
    behind them. The canvas is the user's and stays as it is, with the
    photo fitted in; only a new, untouched editor (no people, no photo,
    the default size) takes the size of the photo.
    """
    try:
        people, size = load(value)
        fresh = (not people and not json.loads(value).get('photo')
                 and size == DEFAULT_SIZE)
    except (ValueError, TypeError):
        people, size, fresh = [], DEFAULT_SIZE, True
    if fresh:
        size = photo.size
    box = fit_box(size, photo.size)
    left, top, right, bottom = box
    for person in found:
        people.append([
            None if p is None else (left + p[0] * (right - left),
                                    top + p[1] * (bottom - top))
            for p in person
        ])
    return dump(people[:MAX_PEOPLE], size, photo, box)


def photo_url(image: Image.Image) -> str:
    """The photo as a small JPEG data URL, to show behind the skeletons."""
    small = image.convert('RGB')
    small.thumbnail((PHOTO_SIDE, PHOTO_SIDE))
    buffer = io.BytesIO()
    small.save(buffer, 'JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(
        buffer.getvalue()).decode()


def _side(value) -> int:
    try:
        return min(MAX_SIDE, max(8, int(value)))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("the size of the canvas is not a number")


def _coordinate(value) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError("a point is not a pair of numbers")
    return min(1.0, max(0.0, float(value)))


def load(value) -> tuple[list[list], tuple[int, int]]:
    """
    (people, (width, height)) from the editor's value, in what
    skeleton.draw takes. Raises ValueError when it isn't a valid value.
    """
    try:
        data = json.loads(value)
        people = data['people']
        size = (_side(data['w']), _side(data['h']))
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError(f"the editor sent something unreadable ({e})")
    if not isinstance(people, list) or len(people) > MAX_PEOPLE:
        raise ValueError(f"at most {MAX_PEOPLE} people")

    result = []
    for person in people:
        if not isinstance(person, list) or len(person) != POINTS:
            raise ValueError(f"a person has {POINTS} points")
        points = []
        for point in person:
            if point is None:
                points.append(None)
            elif isinstance(point, list) and len(point) == 2:
                points.append((_coordinate(point[0]), _coordinate(point[1])))
            else:
                raise ValueError("a point is not a pair of numbers")
        result.append(points)
    return result, size


def render(value: str | None = None) -> gr.HTML:
    """The editor component."""
    return gr.HTML(
        value or blank(512, 768),
        html_template='<div class="pose-editor"></div>',
        css_template=STYLE,
        js_on_load=SCRIPT.read_text(encoding='utf-8'),
        limbs=skeleton.LIMBS,
        colors=skeleton.COLORS,
        label="Pose editor",
    )

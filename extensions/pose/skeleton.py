"""OpenPose skeletons: from body landmarks to the image ControlNet reads."""

from PIL import Image, ImageDraw

# The 18 points of OpenPose, in order: nose, neck, right shoulder, elbow,
# wrist, left shoulder, elbow, wrist, right hip, knee, ankle, left hip,
# knee, ankle, right eye, left eye, right ear, left ear.
# Each is the index of the same point among MediaPipe's 33 landmarks; the
# neck has none, it is the middle of the shoulders.
FROM_LANDMARKS = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27,
                  5, 2, 8, 7]

# The 17 limbs, as pairs of the points above
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
         (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
         (0, 15), (15, 17)]

# OpenPose's colors: point i and limb i share color i
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
          (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
          (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
          (255, 0, 170), (255, 0, 85)]

LIMB_BRIGHTNESS = 0.6  # limbs are drawn darker than the points


def to_openpose(landmarks, min_visibility: float = 0.5) -> list:
    """
    The 18 OpenPose points of MediaPipe's landmarks, each (x, y) in 0..1,
    or None when it is not visible enough.
    """
    def seen(index):
        landmark = landmarks[index]
        if getattr(landmark, 'visibility', 1.0) < min_visibility:
            return None
        return (landmark.x, landmark.y)

    points = []
    for index in FROM_LANDMARKS:
        if index is not None:
            points.append(seen(index))
            continue
        left, right = seen(11), seen(12)
        points.append(
            ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            if left and right else None
        )
    return points


def draw(
    people: list[list], size: tuple[int, int],
    photo: Image.Image | None = None, fade: float = 0.35
) -> Image.Image:
    """
    The skeletons of `people` (a list of to_openpose results) at `size`:
    on black, which is what ControlNet wants, or on a faded copy of the
    photo, to check that it fits them.
    """
    width, height = size
    canvas = Image.new('RGB', size, 'black')
    if photo is not None:
        canvas = Image.blend(canvas, photo.convert('RGB').resize(size), fade)

    pen = ImageDraw.Draw(canvas)
    thickness = max(2, round(min(size) / 100))

    def at(point):
        return (point[0] * width, point[1] * height)

    for points in people:
        for (a, b), color in zip(LIMBS, COLORS):
            if points[a] and points[b]:
                pen.line(
                    [at(points[a]), at(points[b])],
                    fill=tuple(int(c * LIMB_BRIGHTNESS) for c in color),
                    width=thickness * 2,
                )
        for point, color in zip(points, COLORS):
            if point:
                x, y = at(point)
                pen.ellipse(
                    (x - thickness, y - thickness,
                     x + thickness, y + thickness),
                    fill=color,
                )
    return canvas

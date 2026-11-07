"""Pose: turns a photo into an OpenPose skeleton for ControlNet.

The Pose tab adds the people it finds in a photo (or starts from an empty
canvas) to a list of skeletons, lets you move their points, resize them and
the canvas, add or delete skeletons, and saves the
skeleton in OpenPose's colors on black, the control image the OpenPose
ControlNet models read. It is saved in the extension's output folder (its
own button in the Gallery); load it as the control image in a generation
tab. The Gallery has a "Make pose" button for the selected
image. Several people per photo are possible: set 'Max people' in the
tab (its starting value, and what the Gallery button uses, is the
'Max people' setting in the Extensions tab).

MediaPipe needs a 9 MB model that is not part of its package: the tab has
a button to download it, once, from Google's MediaPipe storage.
"""

import gradio as gr
from PIL import Image

from . import detector, editor, skeleton

NAME = 'Pose'
VERSION = '1.0'
AUTHOR = 'daniandtheweb'
URL = 'https://github.com/daniandtheweb/sd.cpp-webui'
DESCRIPTION = (
    "Turns a photo of a person into an OpenPose skeleton image, to use "
    "as a ControlNet control image."
)
REQUIRES = {
    'mediapipe>=1.2.2,<2': "finds the body landmarks of the people in a photo",
}
HOSTS = {
    'storage.googleapis.com': "where Google stores the 9 MB pose model "
                              "(downloaded when you press the button)",
}
API_VERSION = 1

BLACK = "Black (what ControlNet reads)"
FADED = "Over the photo (to check it fits)"


def register(api):
    detector.set_model_dir(api.data_dir())
    api.add_setting(
        'max_people', 1, label="Max people",
        info="How many people to look for in a photo: the starting value "
             "of the Pose tab, and what the Gallery button uses.",
    )
    api.add_output(label='Poses')
    api.add_tab('Pose', lambda: render_tab(api))
    api.add_gallery_action("Make pose", lambda ctx: pose_selected(api, ctx))


def pose_selected(api, ctx):
    """The Gallery action: the skeleton of the selected image."""
    with Image.open(ctx.path) as photo:
        return make_pose(
            api, photo.convert('RGB'), 0.5, BLACK,
            api.get_setting('max_people')
        )[1]


def make_pose(api, image, confidence, background, max_people=1):
    """Detect, draw and save the skeleton: (preview, status)."""
    if image is None:
        return None, "Upload a photo first."

    if not detector.model_ready():
        return None, (
            "The pose model is not downloaded yet: press "
            '"Download the model" first.'
        )

    try:
        found = detector.find_landmarks(image, max_people)
    except detector.NoPerson as e:
        return None, str(e)
    except (Exception, SystemExit) as e:
        # SystemExit: some libraries exit when a backend is missing
        return None, f"Pose detection failed: {type(e).__name__}: {e}"

    people = [skeleton.to_openpose(marks, confidence) for marks in found]
    counts = [sum(point is not None for point in p) for p in people]
    people = [p for p, seen in zip(people, counts) if seen >= 2]
    if not people:
        return None, (
            f"Only {max(counts)} point(s) are visible enough. "
            "Try a lower confidence."
        )

    # What gets saved is always the black one: it is the control image
    pose = skeleton.draw(people, image.size)
    path = api.next_output_path()
    pose.save(path)

    shown = pose
    if background == FADED:
        shown = skeleton.draw(people, image.size, photo=image)
    seen = ", ".join(str(n) for n in counts if n >= 2)
    return shown, (
        f"Saved {path} ({len(people)} person(s), points found: {seen}, "
        "of 18 each)."
    )


def detect_people(image, confidence, max_people, current=None):
    """
    Add the people found in the photo to the ones in the editor (current
    is its value), with the photo behind them.
    """
    if image is None:
        return gr.update(), "Upload a photo first."
    if not detector.model_ready():
        return gr.update(), (
            "The pose model is not downloaded yet: press "
            '"Download the model" first.'
        )
    try:
        found = detector.find_landmarks(image, max_people)
    except detector.NoPerson as e:
        # Nothing added: the photo is still shown, to place people by hand
        return editor.add_people(current, [], image), str(e)
    except (Exception, SystemExit) as e:
        return gr.update(), f"Pose detection failed: {type(e).__name__}: {e}"

    people = [skeleton.to_openpose(marks, confidence) for marks in found]
    people = [p for p in people if sum(pt is not None for pt in p) >= 2]
    return editor.add_people(current, people, image), (
        f"Added {len(people)} person(s). Move the points, then save."
    )


def make_blank(width, height):
    """An empty editor of that size."""
    try:
        return editor.blank(width, height), "Add a person to start."
    except ValueError as e:
        return gr.update(), str(e)


def save_edited(api, value):
    """Draw the people of the editor, save it: (preview, status)."""
    try:
        people, size = editor.load(value)
    except ValueError as e:
        return None, f"Not saved: {e}"
    people = [p for p in people if sum(pt is not None for pt in p) >= 2]
    if not people:
        return None, "Nothing to save: add a person (2 points at least)."
    pose = skeleton.draw(people, size)
    path = api.next_output_path()
    pose.save(path)
    return pose, f"Saved {path} ({len(people)} person(s))."


def model_status() -> str:
    if detector.model_ready():
        return "Pose model: downloaded."
    return (
        "Pose model: **not downloaded**. It is 9 MB, from "
        "storage.googleapis.com (MediaPipe), and downloaded only when you "
        "press the button."
    )


def download_model(api) -> str:
    """The download button: the new status of the model."""
    try:
        detector.download_model(lambda url, name: api.download(url, name))
    except Exception as e:
        return f"Could not download the model: {e}\n\n{model_status()}"
    return model_status()


def render_tab(api):
    gr.Markdown(
        "## Pose\n\n"
        "Add the people of a photo (the ones already there stay), or start "
        "from an empty canvas, then drag the points into place. The "
        "skeletons are listed next to the canvas: click one to select it; "
        "each has its own size slider and delete button, and *Add* makes a "
        "new one. The canvas size is yours, whatever the photo, and the "
        "photo can be hidden. Saving puts "
        "the skeleton image in the **Poses** gallery: load it as the "
        "control image of ControlNet (with an OpenPose model)."
    )
    with gr.Row():
        model = gr.Markdown(model_status())
        download = gr.Button(
            "Download the model", scale=0,
            visible=not detector.model_ready(),
        )

    def download_and_hide():
        status = download_model(api)
        return status, gr.update(visible=not detector.model_ready())

    download.click(download_and_hide, outputs=[model, download])
    with gr.Row():
        with gr.Column(scale=1):
            photo = gr.Image(label='Photo', type='pil')
            confidence = gr.Slider(
                label="Minimum confidence", minimum=0.1, maximum=1.0,
                step=0.05, value=0.5,
                info="Points the detector is less sure about are left out.",
            )
            people = gr.Slider(
                label="Max people", minimum=1, maximum=10, step=1,
                value=api.get_setting('max_people'),
                info="The most prominent first. MediaPipe misses small "
                     "or overlapping people.",
            )
            detect = gr.Button(
                "Add the people of the photo", variant='primary'
            )
            blank = gr.Button("Start from an empty canvas")
        with gr.Column(scale=1):
            canvas = editor.render()
            save = gr.Button("Save the pose", variant='primary')
            result = gr.Image(label="Saved pose", type='pil')
            status = gr.Textbox(label='Status', interactive=False)

    detect.click(
        detect_people, inputs=[photo, confidence, people, canvas],
        outputs=[canvas, status],
    )
    blank.click(
        lambda: make_blank(*editor.DEFAULT_SIZE), outputs=[canvas, status]
    )
    save.click(
        lambda value: save_edited(api, value),
        inputs=[canvas], outputs=[result, status],
    )

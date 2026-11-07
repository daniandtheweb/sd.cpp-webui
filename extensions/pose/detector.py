"""Finds the body of a person in a photo (MediaPipe's pose landmarker).

MediaPipe does not ship the model: it is a 9 MB file that is downloaded
once, when the user asks for it (download_model), to the data folder of
the extension (set_model_dir).
"""

import os
from pathlib import Path

import numpy
from PIL import Image

MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
    'pose_landmarker_full/float16/1/pose_landmarker_full.task'
)
MODEL_FILE = 'pose_landmarker_full.task'
MODEL_PATH = None  # set by set_model_dir

# A real model is megabytes: anything smaller is an error page
MIN_MODEL_BYTES = 1_000_000


class NoPerson(Exception):
    """The photo has no person the detector could find."""


def set_model_dir(folder: str):
    """Where the model is kept: the data folder of the extension."""
    global MODEL_PATH
    MODEL_PATH = Path(folder) / MODEL_FILE


def model_ready() -> bool:
    return MODEL_PATH is not None and MODEL_PATH.is_file()


def download_model(fetch) -> str:
    """
    Download the model if it is missing, with fetch(url, filename): the
    api's download, which saves it in the extension's data folder, where
    the app logs it. Returns a message for the user; raises OSError when
    the download fails.
    """
    if model_ready():
        return "The pose model is already downloaded."
    if MODEL_PATH is None:
        raise OSError("there is no folder for the model")

    print(f"Pose: downloading the model from {MODEL_URL}")
    fetch(MODEL_URL, MODEL_FILE)
    size = MODEL_PATH.stat().st_size
    if size < MIN_MODEL_BYTES:
        MODEL_PATH.unlink()
        raise OSError(f"the download is too small ({size} bytes)")
    return f"Downloaded the pose model ({size / 1e6:.1f} MB)."


def find_landmarks(image: Image.Image, max_people: int = 1) -> list[list]:
    """
    MediaPipe's 33 body landmarks of each person in the image (up to
    max_people), each landmark with x and y in 0..1 and a visibility.
    """
    if not model_ready():
        raise FileNotFoundError("the pose model is not downloaded yet")

    # MediaPipe's C++ side prints its warnings (glog) to the terminal;
    # keep the errors. Read when it is first imported.
    os.environ.setdefault('GLOG_minloglevel', '2')
    import mediapipe as mp  # slow to import, so only when it is needed

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=max(1, int(max_people)),
    )
    picture = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=numpy.ascontiguousarray(image.convert('RGB')),
    )
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as landmarker:
        result = landmarker.detect(picture)

    if not result.pose_landmarks:
        raise NoPerson("No person found in the image.")
    return [list(person) for person in result.pose_landmarks]

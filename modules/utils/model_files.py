"""sd.cpp-webui - utils - model files module"""

import os
import time

from modules.loader import SUPPORTED_EXTENSIONS, get_models
from modules.shared_instance import config

# Model type shown in the Models tab -> config key of its folder
MODEL_TYPES = {
    'Checkpoint': 'ckpt_dir',
    'UNET': 'unet_dir',
    'VAE': 'vae_dir',
    'Text encoder': 'txt_enc_dir',
    'Tokenizer': 'tokenizer_dir',
    'TAESD': 'taesd_dir',
    'LoRA': 'lora_dir',
    'Embeddings': 'emb_dir',
    'PhotoMaker': 'phtmkr_dir',
    'Upscaler': 'upscl_dir',
    'ControlNet': 'cnnet_dir',
    'ADetailer': 'ad_dir',
    'IP-Adapter': 'ipadapter_dir',
    'PuLID': 'pulid_dir',
    'AnimateDiff': 'animatediff_dir'
}
NAME_MAX = 200


def model_folder(model_type: str) -> str | None:
    """The folder of a model type, from the config (None if unknown)."""
    key = MODEL_TYPES.get(model_type)
    return config.get(key) if key else None


def size_text(size: int) -> str:
    """A file size for people: '512 B', '1.5 MB', '6.2 GB'."""
    if size < 1024:
        return f"{size} B"
    for unit in ('KB', 'MB'):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"
    return f"{size / 1024:.1f} GB"


def list_models(model_type: str) -> list[dict]:
    """
    The models of a type: name, size and modified as text, and the same
    as numbers (bytes, time) to sort by.
    """
    folder = model_folder(model_type)
    models = []
    for name in get_models(folder):
        try:
            info = os.stat(os.path.join(folder, name))
        except OSError:
            continue
        models.append({
            'name': name,
            'size': size_text(info.st_size),
            'modified': time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(info.st_mtime)),
            'bytes': info.st_size,
            'time': info.st_mtime
        })
    return models


def model_path(folder: str | None, name: str | None) -> str | None:
    """
    The path of a model file inside folder, or None when the name is empty,
    leaves the folder, is not a model file or does not exist.
    """
    if not folder or not name:
        return None
    root = os.path.realpath(folder)
    path = os.path.realpath(os.path.join(root, str(name)))
    if os.path.commonpath([root, path]) != root:
        return None
    if not path.endswith(SUPPORTED_EXTENSIONS) or not os.path.isfile(path):
        return None
    return path


def rename_model(
    model_type: str, name: str, new_name: str
) -> tuple[str, str | None]:
    """
    Rename a model file in its own folder. The extension is kept (it is
    added when missing) and a file that exists is never replaced.
    Returns the message to show and the model's new name (None when it
    was not renamed).
    """
    path = model_path(model_folder(model_type), name)
    if path is None:
        return "Select a model first.", None

    new_name = str(new_name or "").strip()
    if not new_name:
        return "Type the new name.", None
    if (len(new_name) > NAME_MAX or new_name.startswith('.')
            or any(char in new_name for char in '/\\')):
        return "The name must be a plain file name.", None

    extension = os.path.splitext(path)[1]
    if not new_name.lower().endswith(extension.lower()):
        new_name += extension
    target = os.path.join(os.path.dirname(path), new_name)
    if os.path.lexists(target):
        return f"{new_name} already exists.", None

    try:
        os.rename(path, target)
    except OSError as e:
        return f"Could not rename: {e}", None
    folder = os.path.dirname(str(name).replace("\\", "/"))
    renamed = "/".join(filter(None, [folder, new_name]))
    return f"Renamed to {new_name}.", renamed


def remove_model(model_type: str, name: str) -> tuple[str, bool]:
    """Delete a model file. Returns the message to show and if it went."""
    path = model_path(model_folder(model_type), name)
    if path is None:
        return "Select a model first.", False
    try:
        os.remove(path)
    except OSError as e:
        return f"Could not remove: {e}", False
    return f"Removed {os.path.basename(path)}.", True

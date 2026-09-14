"""sd.cpp-webui - utils - resolution preset management module"""

import os
from typing import List, Tuple

from .file_utils import load_json, save_json

DEFAULT_RESOLUTIONS_PATH = os.path.join('user_data', 'resolutions.json')

BUILTIN_RESOLUTIONS = {
    '512x512': {'width': 512, 'height': 512},
    '768x768': {'width': 768, 'height': 768},
    '1024x1024': {'width': 1024, 'height': 1024},
    '1344x768': {'width': 1344, 'height': 768},
    '768x1344': {'width': 768, 'height': 1344},
    '2048x2048': {'width': 2048, 'height': 2048},
}


class ResolutionManager:
    """
    Handles loading, saving and managing user-defined resolution presets.

    Built-in presets are kept in code and can never be deleted or overridden.
    User presets are stored in a JSON file.
    """

    def __init__(self, resolutions_path: str = None):
        self.resolutions_path = os.getenv(
            'SD_WEBUI_RESOLUTIONS_PATH',
            resolutions_path or DEFAULT_RESOLUTIONS_PATH
        )
        self.resolutions = load_json(self.resolutions_path) or {}
        self._initialize_files()

    def _initialize_files(self):
        """Ensures the resolutions file exists."""
        if not os.path.isfile(self.resolutions_path):
            self.save_resolutions()
            print("Created empty resolutions file.")

    def save_resolutions(self):
        """Saves the current user resolutions dictionary to disk."""
        save_json(self.resolutions_path, self.resolutions)

    def get_resolutions(self) -> List[str]:
        """Returns built-in and user preset names."""
        return sorted(list(BUILTIN_RESOLUTIONS.keys()) + list(self.resolutions.keys()))

    def add_resolution(self, name: str, width: int, height: int) -> bool:
        """
        Adds a new user resolution preset.
        Returns False if the name is empty, a built-in preset,
        or already in use.
        """
        if not name or not str(name).strip():
            return False

        clean_name = str(name).strip()

        if clean_name in BUILTIN_RESOLUTIONS or clean_name in self.resolutions:
            return False

        self.resolutions[clean_name] = {'width': int(width), 'height': int(height)}
        self.save_resolutions()
        return True

    def delete_resolution(self, name: str) -> bool:
        """
        Deletes a user resolution preset.
        Returns False if the preset is missing or a built-in.
        """
        if not name or name in BUILTIN_RESOLUTIONS or name not in self.resolutions:
            return False

        del self.resolutions[name]
        self.save_resolutions()
        return True

    def get_resolution(self, name: str) -> Tuple[int, int] | None:
        """Retrieves a specific resolution as a (width, height) tuple."""
        if not name:
            return None

        resolution = BUILTIN_RESOLUTIONS.get(name) or self.resolutions.get(name)
        if resolution is None:
            return None

        return resolution['width'], resolution['height']

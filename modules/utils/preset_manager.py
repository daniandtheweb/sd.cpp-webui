"""sd.cpp-webui - utils - preset management module"""

import os
from typing import Any, Dict, List, Optional

from .file_utils import load_json, save_json

DEFAULT_PRESETS_PATH = os.path.join('user_data', 'presets.json')


class PresetManager:
    """Handles loading, saving and managing user-defined generation presets."""

    def __init__(self, preset_path: str = None):
        self.presets_path = os.getenv(
            'SD_WEBUI_PRESETS_PATH', preset_path or DEFAULT_PRESETS_PATH
        )
        self.presets = load_json(self.presets_path) or {}
        self._initialize_files()

    def _initialize_files(self):
        """Ensures preset file exists."""
        if not os.path.isfile(self.presets_path):
            self.save_presets()
            print("Created empty presets file.")

    def save_presets(self):
        """Saves the current presets dictionary to disk."""
        save_json(self.presets_path, self.presets)

    def get_presets(self) -> List[str]:
        return sorted(list(self.presets.keys()))

    def add_preset(self, name: str, **kwargs):
        if not name:
            return
        self.presets[name.strip()] = kwargs
        self.save_presets()

    def delete_preset(self, name: str):
        if name in self.presets:
            del self.presets[name]
            self.save_presets()

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        return self.presets.get(name)

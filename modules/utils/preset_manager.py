"""sd.cpp-webui - utils - preset management module"""

import os
from typing import Any, Dict, List, Optional

from .file_utils import load_json, save_json

USER_PRESETS_PATH = os.path.join('user_data', 'presets.json')
DEFAULT_PRESETS_PATH = os.path.join('defaults', 'default_presets.json')


class PresetManager:
    """Handles loading, saving and managing user-defined generation presets."""

    def __init__(
            self, user_preset_path: str = None, default_preset_path: str = None
    ):
        self.user_presets_path = os.getenv(
            'SD_WEBUI_PRESETS_PATH', user_preset_path or USER_PRESETS_PATH
        )
        self.default_presets_path = default_preset_path or DEFAULT_PRESETS_PATH

        self.default_presets = load_json(self.default_presets_path) or {}

        self.user_presets = load_json(self.user_presets_path) or {}

        self._initialize_files()

    def _initialize_files(self):
        """Ensures preset file exists."""
        if not os.path.isfile(self.user_presets_path):
            self.save_presets()
            print("Created empty presets file.")

    def is_default(self, name: str) -> bool:
        """Checks if a preset name belongs to the read-only defaults."""
        return name in self.default_presets

    def save_presets(self):
        """Saves only the user presets dictionary to disk."""
        save_json(self.user_presets_path, self.user_presets)

    def get_presets(self) -> List[str]:
        """Returns a sorted list of all preset names (defaults + user)."""
        all_keys = set(self.default_presets.keys()).union(set(self.user_presets.keys()))
        return sorted(list(all_keys))

    def add_preset(self, name: str, **kwargs):
        if not name:
            return

        name = name.strip()

        # Backend guardrail
        if self.is_default(name):
            print(f"Warning: Attempted to overwrite default preset '{name}'. Blocked by backend.")
            return

        self.user_presets[name] = kwargs
        self.save_presets()

    def delete_preset(self, name: str):
        """Deletes a user preset. Blocks deleting defaults."""
        # Backend guardrail
        if self.is_default(name):
            print(f"Warning: Attempted to delete default preset '{name}'. Blocked by backend.")
            return

        if name in self.user_presets:
            del self.user_presets[name]
            self.save_presets()

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a preset, checking defaults first, then user presets."""
        if name in self.default_presets:
            return self.default_presets.get(name)
        return self.user_presets.get(name)

"""sd.cpp-webui - settings - the config file"""

import json
import os
import shutil
from copy import deepcopy
from typing import Any

from modules.options import registry
from modules.settings.defaults import DEFAULT_SETTINGS

DEFAULT_CONFIG_PATH = os.path.join('user_data', 'config.json')


def _strip_trailing_commas(text: str) -> str:
    """
    Removes the commas that sit right before a closing } or ],
    leaving the ones inside strings alone.
    """
    out = []
    in_string = escaped = False
    pending = None  # index in out of a comma not yet followed by a token
    for ch in text:
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch.isspace():
            out.append(ch)
            continue
        if ch in '}]' and pending is not None:
            out[pending] = ''
        pending = None
        if ch == ',':
            pending = len(out)
        elif ch == '"':
            in_string = True
        out.append(ch)
    return ''.join(out)


class ConfigManager:
    """
    Handles loading, managing and saving the application configuration.
    """

    # The default models ('def_ckpt', ...) and the folder each one is in;
    # the registry knows which settings are model files
    PATH_MAP = registry.default_model_folders()

    def __init__(self, config_path: str | None = None):
        self.config_path = os.getenv(
            'SD_WEBUI_CONFIG_PATH', config_path or DEFAULT_CONFIG_PATH
        )
        self.data = self._load_json(self.config_path) or {}
        self._initialize_files()

    def _load_json(self, file_path: str) -> dict[str, Any]:
        """
        Safely loads a JSON file. A file that only has trailing commas
        (the usual hand-edit slip) is repaired and loaded; any other
        error starts from an empty configuration. Either way the
        original is copied to <file>.broken first, because the config
        is saved over the file afterwards.
        """
        if not os.path.isfile(file_path):
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except OSError as e:
            print(f"Error loading {file_path}: {e}. Using empty configuration.")
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            error = e

        repaired = _strip_trailing_commas(text)
        data = None
        if repaired != text:
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                pass

        backup = file_path + '.broken'
        try:
            shutil.copyfile(file_path, backup)
            saved = f"A copy of the original was saved to {backup}"
        except OSError as copy_error:
            saved = f"Could not back up {file_path}: {copy_error}"

        if isinstance(data, dict):
            print(
                f"{file_path} had a syntax error ({error}); "
                f"removed the trailing comma(s) and loaded it. {saved}"
            )
            return data
        print(f"Error loading {file_path}: {error}. Using empty configuration.")
        print(saved)
        return {}

    def _save_json(self, file_path: str, data: dict[str, Any]):
        """Saves data to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            print(f"Error saving to {file_path}: {e}")

    def _initialize_files(self):
        """Ensures config and prompt files exist and have default values."""
        updated_config = False
        for key, value in DEFAULT_SETTINGS.items():
            if key not in self.data:
                self.data[key] = value
                updated_config = True
        if updated_config:
            self.save_config()
            print("Missing settings added to config file.")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a configuration value.
        - If key is a directory (ends in _dir), checks if folder exists.
        - If key is a file (in PATH_MAP), checks if file exists in the
          mapped dir.
        Returns None if the path/file is missing.
        """
        value = self.data.get(key, default)

        if not isinstance(value, str) or not value:
            return value

        if key.endswith('_dir'):
            if not os.path.exists(value):
                return None
            return value

        if key in self.PATH_MAP:
            dir_key = self.PATH_MAP[key]
            dir_path = self.data.get(dir_key)

            if not dir_path or not isinstance(dir_path, str):
                return None

            full_path = os.path.join(dir_path, value)

            if not os.path.exists(full_path):
                return None
            return value

        return value

    def get_or_create_dir(self, key: str) -> str:
        """Get an output directory setting, creating it if missing.

        Unlike :meth:`get` this never returns ``None`` for a valid
        directory key: a stale/invalid value falls back to
        ``DEFAULT_SETTINGS`` and the directory is created on demand.
        Use this for paths that get files written into, so a corrupt
        config cannot break generation.
        """
        value = self.data.get(key)
        if not isinstance(value, str) or not value:
            value = DEFAULT_SETTINGS.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"No configured or default value for directory key '{key}'"
            )
        os.makedirs(value, exist_ok=True)
        return value

    def save_config(self):
        """Saves the current configuration."""
        self._save_json(self.config_path, self.data)

    def update_settings(self, new_settings: dict[str, Any]):
        """Updates the configuration with new settings and saves."""
        self.data.update(new_settings)
        self.save_config()
        print("Set new defaults completed.")

    def reset_defaults(self):
        """Resets the configuration to factory defaults."""
        self.data = deepcopy(DEFAULT_SETTINGS)
        self.save_config()
        print("Reset defaults completed.")

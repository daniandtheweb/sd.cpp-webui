"""sd.cpp-webui - utils - file utils module"""

import os
import json
import shutil
from typing import Any, Dict


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None


def load_json(file_path: str) -> Dict[str, Any]:
    """Safely loads a JSON file."""
    if not os.path.isfile(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error loading {file_path}: {e}. Using empty dictionary.")
        return {}


def save_json(file_path: str, data: Dict[str, Any]):
    """Saves data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except OSError as e:
        print(f"Error saving to {file_path}: {e}")


def migrate_legacy_configs(config_dir: str = "user_data"):
    """
    Silently migrates legacy configuration files from the root directory
    to the new config directory to ensure backward compatibility.
    """
    os.makedirs(config_dir, exist_ok=True)

    legacy_files = {
        'config.json': os.path.join(config_dir, 'config.json'),
        'prompts.json': os.path.join(config_dir, 'prompts.json'),
        'presets.json': os.path.join(config_dir, 'presets.json')
    }

    for old_path, new_path in legacy_files.items():
        if os.path.isfile(old_path):
            if not os.path.isfile(new_path):
                print(f"Migrating legacy config: {old_path} -> {new_path}")
                shutil.move(old_path, new_path)
            else:
                return

"""sd.cpp-webui - utils - file utils module"""

import os
import json
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

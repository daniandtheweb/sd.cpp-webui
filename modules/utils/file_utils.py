"""sd.cpp-webui - utils - file utils module"""

import os


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None

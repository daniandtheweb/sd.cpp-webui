"""sd.cpp-webui - utils - prompt management module"""

import os
from typing import Tuple, List

from .file_utils import load_json, save_json

DEFAULT_PROMPTS_PATH = 'prompts.json'


class PromptManager:
    """Handles loading, saving and managing user-defined prompts."""

    def __init__(self, prompt_path: str = None):
        self.prompts_path = os.getenv(
            'SD_WEBUI_PROMPTS_PATH', prompt_path or DEFAULT_PROMPTS_PATH
        )
        self.prompts = load_json(self.prompts_path) or {}
        self._initialize_files()

    def _initialize_files(self):
        """Ensures prompt file exists."""
        if not os.path.isfile(self.prompts_path):
            self.save_prompts()
            print("Created empty prompts file.")

    def save_prompts(self):
        """Saves the current prompts dictionary to disk."""
        save_json(self.prompts_path, self.prompts)

    def get_prompts(self) -> List[str]:
        """Returns a list of saved prompts."""
        return sorted(list(self.prompts.keys()))

    def add_prompt(self, name: str, positive: str, negative: str):
        """Adds or updates a prompt."""
        if not name:
            return
        self.prompts[name.strip()] = {
            'positive': positive, 'negative': negative
        }
        self.save_prompts()

    def delete_prompt(self, name: str):
        """Deletes a prompt."""
        if name in self.prompts:
            del self.prompts[name]
            self.save_prompts()

    def get_prompt(self, name: str) -> Tuple[str, str]:
        """Retrieves a specific prompt as two generate strings for Gradio."""
        prompt = self.prompts.get(name, {'positive': '', 'negative': ''})
        return prompt['positive'], prompt['negative']

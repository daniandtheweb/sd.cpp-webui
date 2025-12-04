"""sd.cpp-webui - Configuration module"""

import os
import json
from typing import Dict, Any, List

CURRENT_DIR = os.getcwd()
DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_PROMPTS_PATH = 'prompts.json'
DEFAULT_SETTINGS = {
    'ckpt_dir': os.path.join(CURRENT_DIR, "models/checkpoints/"),
    'unet_dir': os.path.join(CURRENT_DIR, "models/unet/"),
    'vae_dir': os.path.join(CURRENT_DIR, "models/vae/"),
    'txt_enc_dir': os.path.join(CURRENT_DIR, "models/text_encoders/"),
    'emb_dir': os.path.join(CURRENT_DIR, "models/embeddings/"),
    'lora_dir': os.path.join(CURRENT_DIR, "models/loras/"),
    'taesd_dir': os.path.join(CURRENT_DIR, "models/taesd/"),
    'phtmkr_dir': os.path.join(CURRENT_DIR, "models/photomaker/"),
    'upscl_dir': os.path.join(CURRENT_DIR, "models/upscale_models/"),
    'cnnet_dir': os.path.join(CURRENT_DIR, "models/controlnet/"),
    'txt2img_dir': os.path.join(CURRENT_DIR, "outputs/txt2img/"),
    'img2img_dir': os.path.join(CURRENT_DIR, "outputs/img2img/"),
    'imgedit_dir': os.path.join(CURRENT_DIR, "outputs/imgedit/"),
    'any2video_dir': os.path.join(CURRENT_DIR, "outputs/any2video/"),
    'upscale_dir': os.path.join(CURRENT_DIR, "outputs/upscale/"),
    'def_type': "Default",
    'def_sampling': "euler_a",
    'def_steps': 20,
    'def_scheduler': "discrete",
    'def_width': 512,
    'def_height': 512,
    'def_cfg': 7.0,
    'def_predict': "Default",
    'def_flash_attn': False,
    'def_diffusion_conv_direct': False,
    'def_vae_conv_direct': False,
    'def_preview_interval': 1,
    'def_output_scheme': "Sequential",
}


class ConfigManager:
    """
    Handles loading, managing and saving the application configuration.
    """

    def __init__(self, config_path: str = None, prompts_path: str = None):
        self.config_path = os.getenv(
            'SD_WEBUI_CONFIG_PATH', config_path or DEFAULT_CONFIG_PATH
        )
        self.prompts_path = os.getenv(
            'SD_WEBUI_PROMPTS_PATH', prompts_path or DEFAULT_PROMPTS_PATH
        )
        self.data = self._load_json(self.config_path) or {}
        self.prompts = self._load_json(self.prompts_path) or {}
        self._initialize_files()

    def _load_json(self, file_path: str) -> Dict[str, Any]:
        """Safely loads a JSON file."""
        if not os.path.isfile(file_path):
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(
                f"Error loading {file_path}: {e}. Using empty configuration."
            )
            return {}

    def _save_json(self, file_path: str, data: Dict[str, Any]):
        """Saves data to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except OSError as e:
            print(f"Error saving to {file_path}: {e}")

    def _initialize_files(self):
        """Ensures config and prompt files exist and have default values."""
        # Initialize config
        updated_config = False
        for key, value in DEFAULT_SETTINGS.items():
            if key not in self.data:
                self.data[key] = value
                updated_config = True
        if updated_config:
            self.save_config()
            print("Missing settings added to config file.")

        # Initialize prompts file if it doesn't exist
        if not os.path.isfile(self.prompts_path):
            self.save_prompts()
            print("Created empty prompts file")

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value."""
        return self.data.get(key, default)

    def save_config(self):
        """Saves the current configuration."""
        self._save_json(self.config_path, self.data)

    def update_settings(self, new_settings: Dict[str, Any]):
        """Updates the configuration with new settings and saves."""
        self.data.update(new_settings)
        self.save_config()
        print("Set new defaults completed.")

    def reset_defaults(self):
        """Resets the configuration to factory defaults."""
        self.data = DEFAULT_SETTINGS.copy()
        self.save_config()
        print("Reset defaults completed.")

    def get_prompts(self) -> List[str]:
        """Returns a list of saved prompts."""
        return sorted(list(self.prompts.keys()))

    def save_prompts(self):
        """Saves the current prompts dictionary to disk."""
        self._save_json(self.prompts_path, self.prompts)

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

    def get_prompt(self, name: str) -> tuple[str, str]:
        """Retrieves a specific prompt as two separate strings for Gradio."""
        prompt = self.prompts.get(name, {'positive': '', 'negative': ''})
        return prompt['positive'], prompt['negative']

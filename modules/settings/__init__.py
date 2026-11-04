"""sd.cpp-webui - saved settings

What the user keeps between runs, in user_data/config.json:

    defaults.py   what a fresh config starts with: the folders and the
                  default of every setting
    store.py      ConfigManager, which reads and writes the config file

The words (option, setting, default, config) are explained in
docs/terminology.md.
"""

from modules.settings.defaults import DEFAULT_SETTINGS, DIR_DEFAULTS
from modules.settings.store import ConfigManager

__all__ = ['ConfigManager', 'DEFAULT_SETTINGS', 'DIR_DEFAULTS']

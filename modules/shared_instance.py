"""sd.cpp-webui - Shared Instance Helper"""

import sys

from modules.utils.file_utils import migrate_legacy_configs
from modules.settings import ConfigManager
from modules.utils.sd_interface import (
    SDChoicesCache, exe_name
)
from modules.utils.subprocess_manager import SubprocessManager
import modules.utils.queue as queue_manager
from modules.utils.ui_state import ModelState
from modules.utils.prompt_manager import PromptManager
from modules.utils.preset_manager import PresetManager
from modules.utils.resolution_manager import ResolutionManager


current_mode = "server" if "--server" in sys.argv else "cli"


def _find_exe(mode):
    """
    The executable of a mode. The one the app runs in is required; the
    other one is optional (None when missing), so an install with only
    sd-cli or only sd-server works.
    """
    try:
        return exe_name(mode)
    except RuntimeError:
        if mode == current_mode:
            raise
        return None


SD_CLI = _find_exe("cli")

SD_SERVER = _find_exe("server")

# The executable of the mode the app runs in
SD_EXE = SD_SERVER if current_mode == "server" else SD_CLI


class ServerState:
    def __init__(self):
        self.running = False
        self.latest_update = {}
        self.ip = ""
        self.port = ""
        self.last_generation_stats = ""
        self.seed = ""


migrate_legacy_configs()

config = ConfigManager()

sd_choices = SDChoicesCache(mode=current_mode)

subprocess_manager = SubprocessManager()

queue_manager.start_worker()

server_state = ServerState()

model_state = ModelState()

prompt_manager = PromptManager()

preset_manager = PresetManager()

resolution_manager = ResolutionManager()

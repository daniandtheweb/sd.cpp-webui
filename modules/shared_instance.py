"""sd.cpp-webui - Shared Instance Helper"""

import sys

from modules.config import ConfigManager
from modules.utils.sd_interface import (
    SDOptionsCache, exe_name
)
from modules.utils.utility import SubprocessManager
import modules.utils.queue as queue_manager


SD_CLI = exe_name("cli")

SD_SERVER = exe_name("server")


class ServerState:
    def __init__(self):
        self.running = False
        self.latest_update = {}
        self.last_generation_stats = ""
        self.seed = ""


config = ConfigManager()

current_mode = "server" if "--server" in sys.argv else "cli"

sd_options = SDOptionsCache(mode=current_mode)

subprocess_manager = SubprocessManager()

queue_manager.start_worker()

server_state = ServerState()

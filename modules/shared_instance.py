"""sd.cpp-webui - Shared Instance Helper"""

from modules.config import ConfigManager
from modules.utils.sd_interface import (
    SDOptionsCache, exe_name
)
from modules.utils.utility import SubprocessManager
import modules.utils.queue as queue_manager


SD_CLI = exe_name("cli")

SD_SERVER = exe_name("server")

config = ConfigManager()

sd_options = SDOptionsCache()

subprocess_manager = SubprocessManager()

queue_manager.start_worker()

"""sd.cpp-webui - Interface for the sdcpp executable"""

import os
import re
import json
import shutil
import hashlib
import subprocess


def exe_name():
    """Returns the stable-diffusion executable name"""
    if os.name == "nt":
        return "sd.exe"
    else:
        return "./sd"


class SDOptionsCache:
    """
    Class to load and cache stable-diffusion.cpp command options synchronously.

    This class provides a robust and efficient way to retrieve and cache
    command-line options from the stable-diffusion.cpp executable's helpers
    output.
    It uses file hashing to ensure the cache is always in sync with the binary.

    Attributes:
        SD_PATH: Full path to the SD binary.
        SD: Executable name for subprocess calls (remains as before).
        _CACHE_FILE: JSON file path to store cached options and binary hash.
        _OPTIONS: List of SD command-line options to cache.
        _help_cache: Dictionary holding cached option values.
    """

    def __init__(self, first_run=False):
        """
        Initializes the SDOptionsCache and prepares the cache.
        """
        self.SD = exe_name()
        self.SD_PATH = self._resolve_sd_path()

        self._CACHE_FILE = "options_cache.json"
        self._OPTIONS = ["--sampling-method", "--scheduler", "--preview",
                         "--type", "--prediction"]
        self._help_cache = {}

        self._load_help_text_sync()

    def _resolve_sd_path(self):
        """Return full path to SD binary across operating systems."""
        sd_path = shutil.which(self.SD)
        if sd_path:
            return sd_path
        fallback = os.path.join(os.getcwd(), self.SD)
        if os.path.isfile(fallback):
            return fallback
        return None

    def _hash_file(self, path):
        """Compute SHA256 hash of a file, or return None if not accessible."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (IOError, FileNotFoundError) as e:
            print(f"Failed to hash file {path}: {e}")
            return None

    def _run_and_cache_help(self, sd_hash):
        """Runs the --help command, parses options, and caches them."""
        help_cache = {}
        try:
            process = subprocess.Popen(
                [self.SD, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            found_options = set()

            while True:
                line = process.stdout.readline()
                if not line:
                    break

                for option in self._OPTIONS:
                    if option in found_options:
                        continue

                    match = re.search(
                        fr"{re.escape(option)}.*\{{([^\}}]+)\}}", line
                    )
                    if match:
                        help_cache[option] = [
                            v.strip() for v in match.group(1).split(",")
                        ]
                        found_options.add(option)
                        continue

                if len(found_options) == len(self._OPTIONS):
                    process.kill()
                    break

            process.stdout.close()
            process.stderr.close()
            process.wait()

        except FileNotFoundError:
            print(
                f"SD binary not found at {self.SD_PATH}. "
                f"Cannot get options."
            )
        except Exception as e:
            print(f"Failed to run SD --help command: {e}")

        self._help_cache = {
            opt: help_cache.get(opt, []) for opt in self._OPTIONS
        }

        try:
            with open(self._CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sd_hash": sd_hash, "options": self._help_cache
                    }, f, indent=2
                )
        except (IOError, json.JSONEncodeError) as e:
            print(f"Failed to write to cache file: {e}")

    def _load_help_text_sync(self, force_refresh=False):
        """Synchronously load SD --help output and cache options to JSON."""
        sd_hash = self._hash_file(self.SD_PATH)

        if force_refresh:
            self._run_and_cache_help(sd_hash)
            return

        if os.path.exists(self._CACHE_FILE):
            try:
                with open(self._CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("sd_hash") == sd_hash:
                        self._help_cache = data.get("options", {})
                        if self._help_cache:
                            return  # Successfull load from cache
            except (IOError, json.JSONDecodeError) as e:
                print(
                    f"Error reading cache file: {e}. "
                    f"Re-running help command."
                )

        # Fallback if cache is missing, invalid, or out of date
        self._run_and_cache_help(sd_hash)

    def _parse_help_option(self, option_name: str):
        """Return cached values for a given SD --help option."""
        return self._help_cache.get(option_name, [])

    def refresh(self):
        """
        Public method to force a refresh of the options from the executable.
        """
        self._load_help_text_sync(force_refresh=True)

    def get_opt(self, option: str):
        """Public getter to return values for a named option.

        Args:
            option: Name of the option ('samplers', 'schedulers', 'previews',
                    'prediction').

        Returns:
            List of available values for the option.

        Raises:
            ValueError: If unknown option name is provided.
        """
        option_map = {
            "samplers": "--sampling-method",
            "schedulers": "--scheduler",
            "previews": "--preview",
            "prediction": "--prediction"
        }
        if option not in option_map:
            raise ValueError(
                f"Unknown option '{option}'. "
                f"Valid options are: {list(option_map.keys())}"
            )
        return self._parse_help_option(option_map[option])

"""sd.cpp-webui - Interface for the sdcpp executable"""

import os
import re
import json
import shutil
import hashlib
import subprocess


def exe_name(mode="cli"):
    """
    Returns the stable-diffusion executable name.
    Prioritizes 'sd-cli' over 'sd', and checks both PATH and current directory.
    Verifies the binary can be executed by running <binary> --version.
    Accumulates errors for failed candidates, only exits if all fail.
    """
    if mode == "server":
        candidates = ["sd-server"]
    elif mode == "cli":
        candidates = ["sd-cli", "sd"]

    if os.name == "nt":
        candidates = [f"{c}.exe" for c in candidates]

    failed_candidates = []

    for cand in candidates:
        executable_path = shutil.which(cand)

        # Check if executable exists in current directory
        if not executable_path:
            local_path = os.path.join(os.getcwd(), cand)
            if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
                executable_path = local_path
                is_local = True

        if executable_path:
            try:
                subprocess.run(
                    [executable_path, "--version"],
                    capture_output=True, text=True, check=True
                )

                if is_local and os.name != "nt":
                    return f"./{cand}"

                return cand

            except subprocess.CalledProcessError as e:
                failed_candidates.append(
                    (
                     f"{executable_path} cannot be executed. " +
                     f"Stdout: {e.stdout.strip()}. Stderr: {e.stderr.strip()}"
                    )
                )
            except Exception as e:
                failed_candidates.append(
                    (
                     f"{executable_path} cannot be executed. " +
                     f"Exception: {str(e)}"
                    )
                )

    # If any candidates were found but failed execution, warn about them
    for error in failed_candidates:
        print(f"Warning: {error}")

    # If no candidates were found or all failed, raise an error
    raise RuntimeError(
        f"Could not find valid executable for mode '{mode}' "
        f"(tried: {', '.join(candidates)}) in PATH or current directory."
    )


class SDOptionsCache:
    """
    Class to load and cache stable-diffusion.cpp command options synchronously.

    This class provides a robust and efficient way to retrieve and cache
    command-line options from the stable-diffusion.cpp executable's helpers
    output.
    It uses file hashes of both the sd binary and this script to ensure the
    cache is always in sync.

    Attributes:
        SD_PATH: Full path to the SD binary.
        SCRIPT_PATH: Full path to this Python script.
        SD: Executable name for subprocess calls (remains as before).
        _CACHE_FILE: JSON file path to store cached options and
                     dependency hashes.
        _OPTIONS: List of SD command-line options to cache.
        _help_cache: Dictionary holding cached option values.
    """

    def __init__(self, mode="cli", first_run=False):
        """
        Initializes the SDOptionsCache and prepares the cache.
        """
        try:
            self.SD_PATH = exe_name(mode=mode)
            self.SD = os.path.basename(self.SD_PATH)
        except RuntimeError as e:
            print(f"SDOptionsCache Initialization Error: {e}")
            self.SD_PATH = None
            self.SD = None

        self.SCRIPT_PATH = os.path.abspath(__file__)

        self._CACHE_FILE = "options_cache.json"
        self._OPTIONS = ["--sampling-method", "--scheduler", "--preview",
                         "--type", "--prediction", "--rng", "--sampler-rng"]
        self._help_cache = {}

        self._load_help_text_sync()

    def _hash_file(self, path):
        """Compute SHA256 hash of a file, or return None if not accessible."""
        if not path or not os.path.exists(path):
            return None

        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()
        except (IOError, FileNotFoundError) as e:
            print(f"Failed to hash file {path}: {e}")
            return None

    def _run_and_cache_help(self):
        """Runs the --help command, parses options, and caches them."""
        if not self.SD_PATH:
            print("Cannot run --help: Executable path is None.")
            return

        help_cache = {}
        try:
            process = subprocess.run(
                [self.SD_PATH, "--help"],
                capture_output=True,
                text=True,
                check=False
            )
            help_text = process.stdout

            for option in self._OPTIONS:
                match = re.search(
                    fr"^\s*{re.escape(option)}\s.*?\[([^\]]+)\]",
                    help_text,
                    re.MULTILINE
                )
                if match:
                    values_str = match.group(1).replace('\n', ' ')
                    help_cache[option] = [
                        v.strip() for v in values_str.split(',') if v.strip()
                    ]

        except Exception as e:
            print(f"Failed to run SD --help command: {e}")

        self._help_cache = {
            opt: help_cache.get(opt, []) for opt in self._OPTIONS
        }

        sd_hash = self._hash_file(self.SD_PATH)
        script_hash = self._hash_file(self.SCRIPT_PATH)

        try:
            with open(self._CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sd_hash": sd_hash,
                        "script_hash": script_hash,
                        "options": self._help_cache
                    }, f, indent=2
                )
        except (IOError, TypeError, ValueError) as e:
            print(f"Failed to write to cache file: {e}")

    def _load_help_text_sync(self, force_refresh=False):
        """Synchronously load SD --help output and cache options to JSON."""
        if (
            force_refresh
            or not self.SD_PATH
            or not os.path.exists(self.SD_PATH)
        ):
            self._run_and_cache_help()
            return

        cache_valid = False
        if os.path.exists(self._CACHE_FILE):
            try:
                with open(self._CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)

                current_sd_hash = self._hash_file(self.SD_PATH)
                current_script_hash = self._hash_file(self.SCRIPT_PATH)

                if (data.get("sd_hash") == current_sd_hash and
                        data.get("script_hash") == current_script_hash):
                    self._help_cache = data.get("options", {})
                    if self._help_cache:
                        cache_valid = True

            except (IOError, json.JSONDecodeError) as e:
                print(
                    f"Error reading cache file: {e}. Rebuilding cache."
                )

        if not cache_valid:
            self._run_and_cache_help()

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
                    'prediction', 'rng', 'sampler_rng').

        Returns:
            List of available values for the option.

        Raises:
            ValueError: If unknown option name is provided.
        """
        option_map = {
            "samplers": "--sampling-method",
            "schedulers": "--scheduler",
            "previews": "--preview",
            "prediction": "--prediction",
            "rng": "--rng",
            "sampler_rng": "--sampler-rng"
        }
        if option not in option_map:
            raise ValueError(
                f"Unknown option '{option}'. "
                f"Valid options are: {list(option_map.keys())}"
            )

        parsed_options = self._parse_help_option(option_map[option])

        if not parsed_options:
            fallbacks = {
                "samplers": [
                    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a",
                    "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm",
                    "ddim_trailing", "tcd", "res_multistep", "res_2s"
                ],
                "schedulers": [
                    "discrete", "karras", "exponential", "ays", "gits",
                    "smoothstep", "sgm_uniform", "simple", "kl_optimal",
                    "lcm", "bong_tangent"
                ],
                "previews": [
                    "none", "proj", "tae", "vae"
                ],
                "prediction": [
                    "eps", "v", "edm_v", "sd3_flow", "flux_flow",
                    "flux2_flow"
                ],
                "rng": [
                    "std_default", "cuda", "cpu"
                ],
                "sampler_rng": [
                    "std_default",
                    "cuda",
                    "cpu"
                ],
            }
            return fallbacks.get(option, [])

        return parsed_options

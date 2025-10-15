"""sd.cpp-webui - Utility module"""

import os
import re
import sys
import shutil
import subprocess
import hashlib
import json
from PIL import Image

import gradio as gr


class SubprocessManager:
    """Class to manage subprocess execution and control.

    Attributes:
        process: The currently running subprocess,
                 or None if no subprocess is active.
    """

    def __init__(self):
        """Initializes the SubprocessManager with no active subprocess."""
        self.process = None
        self.STATS_REGEX = re.compile(r"completed, taking ([\d.]+)s")
        self.TOTAL_TIME_REGEX = re.compile(r"completed in ([\d.]+)s")
        self.ETA_REGEX = re.compile(r'(\d+)/(\d+)\s*-\s*([\d.]+)(s/it|it/s)')
        self.SIMPLE_REGEX = re.compile(r'(\d+)/(\d+)')

    def _parse_final_stats(self, line, final_stats):
        """
        Parses a line for final summary stats and updates the stats dictionary.
        """
        if 'loading tensors completed' in line:
            match = self.STATS_REGEX.search(line)
            if match:
                final_stats['tensor_load_time'] = f"{match.group(1)}s"
        elif 'sampling completed' in line:
            match = self.STATS_REGEX.search(line)
            if match:
                final_stats['sampling_time'] = f"{match.group(1)}s"
        elif 'decode_first_stage completed' in line:
            match = self.STATS_REGEX.search(line)
            if match:
                final_stats['decoding_time'] = f"{match.group(1)}s"
        elif 'generate_image completed' in line:
            match = self.TOTAL_TIME_REGEX.search(line)
            if match:
                final_stats['total_time'] = f"{match.group(1)}s"

    def _parse_progress_update(self, line, final_stats):
        """Parses a progress bar line and returns a dictionary for the UI."""
        eta_match = self.ETA_REGEX.search(line)
        if eta_match:
            (current_step, total_steps,
             speed_value, speed_unit) = eta_match.groups()
            final_stats['last_speed'] = (
                    f"{float(speed_value):.2f} {speed_unit}"
            )

            current_step, total_steps = map(int, [current_step, total_steps])
            speed_value = float(speed_value)

            phase_fraction = current_step / total_steps
            steps_remaining = total_steps - current_step
            eta_seconds = 0

            if speed_unit == 's/it':
                eta_seconds = int(steps_remaining * speed_value)
            elif speed_unit == 'it/s' and speed_value > 0:
                eta_seconds = int(steps_remaining / speed_value)

            if eta_seconds < 60:
                eta_str = f"{eta_seconds}s"
            elif eta_seconds < 3600:  # Less than an hour
                minutes = eta_seconds // 60
                seconds = eta_seconds % 60
                eta_str = f"{minutes:02}:{seconds:02}"
            else:  # An hour or more
                hours = eta_seconds // 3600
                minutes = (eta_seconds % 3600) // 60
                seconds = eta_seconds % 60
                eta_str = f"{hours:02}:{minutes:02}:{seconds:02}"

            return {
                "percent": int(phase_fraction * 100),
                "status": (
                    f"Speed: {final_stats['last_speed']} | "
                    f"ETA: {eta_str}"
                )
            }

        # Fallback for progress lines without ETA info
        simple_match = self.SIMPLE_REGEX.search(line)
        if simple_match:
            current_step, total_steps = map(int, simple_match.groups())
            phase_fraction = current_step / total_steps
            return {
                "percent": int(phase_fraction * 100),
                "status": f"Step: {current_step}/{total_steps}"
            }
        return {}

    def run_subprocess(self, command, env=None):
        """
        Runs a subprocess, captures its output, and yields UI updates.
        This main method is now much simpler and delegates parsing to helpers.
        """
        phase = "Initializing"
        last_was_progress = False
        final_stats = {}

        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                env=env,
                errors='replace'
            ) as self.process:

                for output_line in self.process.stdout:
                    output_line = output_line.rstrip()

                    self._parse_final_stats(output_line, final_stats)

                    if 'loading model' in output_line:
                        phase = "Loading Model"
                    elif 'sampling using' in output_line:
                        phase = "Sampling"
                    elif 'upscaling from' in output_line:
                        phase = "Upscaling"

                    if "|" in output_line and "/" in output_line:
                        if phase in ["Sampling", "Upscaling"]:
                            update_data = self._parse_progress_update(
                                output_line,
                                final_stats
                            )
                            if update_data:
                                yield update_data

                        sys.stdout.write(f"\r{output_line}")
                        sys.stdout.flush()
                        last_was_progress = True
                    else:
                        if last_was_progress:
                            print("\n")
                            last_was_progress = False
                        print(output_line)

        finally:
            if last_was_progress:
                print("\n")

            if self.process and self.process.returncode != 0:
                print("Subprocess terminated.")

            self.process = None

        yield {"final_stats": final_stats}

    def kill_subprocess(self):
        """Terminates the currently running subprocess, if any.

        This method sets the subprocess attribute to None after termination
        and prints a message indicating whether a subprocess was running.
        """
        if self.process is not None:
            self.process.terminate()
        else:
            print("No subprocess running.")


subprocess_manager = SubprocessManager()


def exe_name():
    """Returns the stable-diffusion executable name"""
    if os.name == "nt":
        return "sd.exe"
    else:
        return "./sd"


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None





def switch_sizes(height, width):
    return (width, height)


def size_extractor(image):
        try:
            with Image.open(image) as img:
                width, height = img.size
        except Exception:
            width, height = None, None
        return (
            width, height
        )


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

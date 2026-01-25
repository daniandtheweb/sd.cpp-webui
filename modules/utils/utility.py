"""sd.cpp-webui - Utility module"""

import os
import re
import sys
import subprocess

import gradio as gr


class SubprocessManager:
    """
    Class to manage subprocess execution and control.

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

    def _determine_phase(self, line):
        """
        Determines the current phase of
        the subprocess based on the output line.
        """
        if 'loading model' in line or 'loading diffusion model' in line:
            return "Loading Model"
        elif 'sampling using' in line:
            return "Sampling"
        elif 'upscaling from' in line:
            return "Upscaling"
        return None

    def _is_progress_line(self, line, phase):
        """
        Checks if a line is a progress bar line.
        """
        return (
            phase in ["Loading Model", "Sampling", "Upscaling"] and
            "|" in line and
            "/" in line
        )

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

    def _process_line(self, raw_line, clean_line, phase,
                      final_stats, last_was_progress):
        """
        Processes output with visual padding around progress bars.
        """
        self._parse_final_stats(clean_line, final_stats)

        display_line = raw_line.rstrip()
        if not display_line:
            return None, last_was_progress

        is_progress = self._is_progress_line(clean_line, phase)

        if is_progress:
            if not last_was_progress:
                sys.stdout.write("\n")

            update_data = self._parse_progress_update(clean_line, final_stats)
            if update_data:
                sys.stdout.write(f"\r{display_line}")
                sys.stdout.flush()
                return update_data, True
        else:
            if last_was_progress:
                sys.stdout.write("\n\n")

            print(display_line)
            return None, False

        return None, last_was_progress

    def _stream_output(self):
        """
        Generates raw strings from the subprocess stdout,
        breaking at line endings or progress updates."""
        buffer = bytearray()
        for byte in iter(lambda: self.process.stdout.read(1), b''):
            buffer.extend(byte)

            is_line_end = byte in (b'\r', b'\n')
            is_progress_end = (
                    len(buffer) > 4 and
                    byte in (b's', b't') and
                    (buffer.endswith(b'it/s') or buffer.endswith(b's/it'))
            )

            if is_line_end or is_progress_end:
                yield buffer.decode('utf-8', errors='replace')
                buffer.clear()

        if buffer:
            yield buffer.decode('utf-8', errors='replace')

    def run_subprocess(self, command, env=None):
        """
        Runs a subprocess and yields UI updates with reduced complexity.
        """
        phase = "Initializing"
        final_stats = {}
        last_was_progress = False
        last_processed_content = ""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        try:
            with subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0, env=env
            ) as self.process:

                for raw_line in self._stream_output():
                    clean_line = ansi_escape.sub('', raw_line)

                    if (
                        clean_line == last_processed_content or
                        not clean_line.strip()
                    ):
                        continue

                    new_phase = self._determine_phase(clean_line)
                    if new_phase:
                        phase = new_phase

                    update_data, last_was_progress = self._process_line(
                        raw_line, clean_line, phase,
                        final_stats, last_was_progress
                    )
                    last_processed_content = clean_line

                    if update_data:
                        yield update_data

        finally:
            if last_was_progress:
                print("\n")
            if self.process and self.process.returncode != 0:
                print("Subprocess terminated.")
            self.process = None

        yield {"final_stats": final_stats}

    def kill_subprocess(self):
        """
        Terminates the currently running subprocess, if any.

        This method sets the subprocess attribute to None after termination
        and prints a message indicating whether a subprocess was running.
        """
        if self.process is not None:
            self.process.terminate()
        else:
            print("No subprocess running.")


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None

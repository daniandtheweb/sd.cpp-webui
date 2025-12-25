"""sd.cpp-webui - Utility module"""

import os
import re
import sys
import subprocess

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
        """
        phase = "Initializing"
        final_stats = {}
        last_was_progress = False
        
        # Regex to strip ANSI color codes ONLY for the internal parser.
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        try:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0, 
                env=env,
            ) as self.process:
                
                buffer = bytearray()
                
                while True:
                    byte = self.process.stdout.read(1)
                    
                    if not byte and self.process.poll() is not None:
                        break
                    if not byte:
                        continue

                    if byte == b'\r' or byte == b'\n':
                        try:
                            output_line = buffer.decode('utf-8', errors='replace')
                            clean_line = ansi_escape.sub('', output_line)
                            self._parse_final_stats(clean_line, final_stats)

                            if 'loading model' in clean_line:
                                phase = "Loading Model"
                            elif 'sampling using' in clean_line:
                                phase = "Sampling"
                            elif 'upscaling from' in clean_line:
                                phase = "Upscaling"

                            if "|" in clean_line and "/" in clean_line:
                                if phase in ["Sampling", "Upscaling"]:
                                    update_data = self._parse_progress_update(
                                        clean_line,
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

                        except Exception:
                            pass 
                        
                        buffer = bytearray()
                    else:
                        buffer.extend(byte)

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


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None

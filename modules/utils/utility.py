"""sd.cpp-webui - Utility module"""

import os
import re
import subprocess
import pty
import select
import errno

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
        self.master_fd = None # Keep track of the PTY master
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
        Runs a subprocess using a PTY (Pseudo-Terminal), captures its output, 
        and yields UI updates.
        """
        phase = "Initializing"
        final_stats = {}
        
        # Open the pseudo-terminal pair
        self.master_fd, slave_fd = pty.openpty()

        try:
            self.process = subprocess.Popen(
                command,
                stdout=slave_fd,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                env=env,
                errors='replace',
                close_fds=True
            )
            
            os.close(slave_fd)

            text_buffer = ""

            while True:
                reads, _, _ = select.select([self.master_fd], [], [], 0.1)

                if self.master_fd in reads:
                    try:
                        output_bytes = os.read(self.master_fd, 1024)
                        if not output_bytes:
                            break # EOF
                    except OSError as e:
                        if e.errno == errno.EIO:
                            break # Linux PTY EOF
                        raise

                    # Decode and print to real terminal IMMEDIATELY
                    chunk = output_bytes.decode('utf-8', errors='replace')
                    print(chunk, end='', flush=True)

                    # Add to buffer for parsing
                    text_buffer += chunk

                    # Process the buffer line by line (splitting on \n OR \r)
                    while True:
                        # Find the first newline or carriage return
                        match = re.search(r'[\r\n]', text_buffer)
                        if not match:
                            break # No full line yet, wait for next chunk
                        
                        # Extract the line
                        end_pos = match.end()
                        line = text_buffer[:match.start()].strip()
                        text_buffer = text_buffer[end_pos:] # Remove processed part

                        if not line:
                            continue

                        self._parse_final_stats(line, final_stats)

                        if 'loading model' in line:
                            phase = "Loading Model"
                        elif 'sampling using' in line:
                            phase = "Sampling"
                        elif 'upscaling from' in line:
                            phase = "Upscaling"

                        if "|" in line and "/" in line:
                            if phase in ["Sampling", "Upscaling"]:
                                update_data = self._parse_progress_update(
                                    line,
                                    final_stats
                                )
                                if update_data:
                                    yield update_data
                
                if self.process.poll() is not None and not reads:
                    break

        finally:
            if self.master_fd:
                try:
                    os.close(self.master_fd)
                except OSError:
                    pass
                self.master_fd = None

            if self.process and self.process.returncode != 0:
                print("\nSubprocess terminated.")

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

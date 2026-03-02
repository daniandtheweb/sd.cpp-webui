"""sd.cpp-webui - subprocess management module"""

import re
import subprocess

from modules.utils.log_parser import LogParser


class SubprocessManager:
    """
    Class to manage subprocess execution and control.

    Attributes:
        process: The currently running subprocess
        or None if no subprocess is active.
    """

    def __init__(self):
        self.process = None
        self.parser = LogParser()

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
        Runs a subprocess and yields UI updates.
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

                    new_phase = self.parser.determine_phase(clean_line)
                    if new_phase:
                        phase = new_phase

                    update_data, last_was_progress = self.parser.process_line(
                        raw_line, clean_line, phase,
                        final_stats, last_was_progress
                    )
                    last_processed_content = clean_line

                    if update_data:
                        yield update_data
                        if "final_stats" in update_data:
                            final_stats.clear()

        finally:
            if last_was_progress:
                print("\n")
            if self.process and self.process.returncode != 0:
                print("Subprocess terminated.")
            self.process = None

        if final_stats:
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

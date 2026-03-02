"""sd.cpp-webui - utils - log parsing module"""

import re
import sys


class LogParser:
    """
    Class to parse stdout logs and calculate progress and ETAs.
    """

    def __init__(self):
        self.STATS_REGEX = re.compile(r"completed, taking ([\d.]+)s")
        self.TOTAL_TIME_REGEX = re.compile(r"completed in ([\d.]+)s")
        self.ETA_REGEX = re.compile(r'(\d+)/(\d+)\s*-\s*([\d.]+)(s/it|it/s)')
        self.SIMPLE_REGEX = re.compile(r'(\d+)/(\d+)')
        self.SEED_REGEX = re.compile(r"generating image:.*?seed\s+(\d+)")

    def _is_progress_line(self, line, phase):
        """
        Checks if a line is a progress bar line.
        """
        return (
            phase in ["Loading Model", "Sampling", "Upscaling"] and
            "|" in line and
            "/" in line
        )

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

        elif 'seed' in line and 'generating image:' in line:
            match = self.SEED_REGEX.search(line)
            if match:
                from modules.shared_instance import server_state
                server_state.seed = int(match.group(1))

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

    def process_line(self, raw_line, clean_line, phase,
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
            if 'generate_image completed' in clean_line:
                return {"final_stats": dict(final_stats)}, False
            return None, False

        return None, last_was_progress

    def determine_phase(self, line):
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

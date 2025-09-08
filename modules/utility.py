"""sd.cpp-webui - Utility module"""

import os
import re
import sys
import shutil
import subprocess

import gradio as gr

from modules.config import (
    def_ckpt, def_unet, def_ckpt_vae, def_unet_vae, def_clip_g, def_clip_l,
    def_t5xxl
)


class ModelState:
    """Class to manage the state of model parameters for the application.

    Attributes:
        bak_ckpt_model: The backup checkpoint model.
        bak_unet_model: The backup UNET model.
        bak_ckpt_vae: The backup checkpoint VAE model.
        bak_unet_vae: The backup UNET VAE model.
        bak_clip_g: The backup CLIP_G model.
        bak_clip_l: The backup CLIP_L model.
        bak_t5xxl: The backup T5-XXL model.
        bak_nprompt: The backup negative prompt.
    """

    def __init__(self):
        """Initializes the ModelState with default values from the
        configuration."""
        self.bak_ckpt_model = def_ckpt
        self.bak_unet_model = def_unet
        self.bak_ckpt_vae = def_ckpt_vae
        self.bak_unet_vae = def_unet_vae
        self.bak_clip_g = def_clip_g
        self.bak_clip_l = def_clip_l
        self.bak_t5xxl = def_t5xxl
        self.bak_nprompt = None

    def update(self, **kwargs):
        """Generic method to update state variables.

        Args:
            kwargs: Key-value pairs of attributes to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{key} is not a valid attribute of ModelState.")

    def bak_ckpt_tab(self, ckpt_model, ckpt_vae, nprompt):
        """Updates the state with values from the checkpoint tab."""
        self.update(
            bak_ckpt_model=ckpt_model,
            bak_ckpt_vae=ckpt_vae,
            bak_nprompt=nprompt
        )

    def bak_unet_tab(self, unet_model, unet_vae, clip_g, clip_l, t5xxl):
        """Updates the state with values from the UNET tab."""
        self.update(
            bak_unet_model=unet_model,
            bak_unet_vae=unet_vae,
            bak_clip_g=clip_g,
            bak_clip_l=clip_l,
            bak_t5xxl=t5xxl
        )


class SubprocessManager:
    """Class to manage subprocess execution and control.

    Attributes:
        process: The currently running subprocess,
                 or None if no subprocess is active.
    """

    def __init__(self):
        """Initializes the SubprocessManager with no active subprocess."""
        self.process = None
        # --- Patterns are now defined once as class attributes ---
        self.STATS_REGEX = re.compile(r"completed, taking ([\d.]+)s")
        self.TOTAL_TIME_REGEX = re.compile(r"completed in ([\d.]+)s")
        self.ETA_REGEX = re.compile(r'(\d+)/(\d+)\s*-\s*([\d.]+)(s/it|it/s)')
        self.SIMPLE_REGEX = re.compile(r'(\d+)/(\d+)')

    def _parse_final_stats(self, line, final_stats):
        """Parses a line for final summary stats and updates the stats dictionary."""
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
            current_step, total_steps, speed_value, speed_unit = eta_match.groups()
            final_stats['last_speed'] = f"{float(speed_value):.2f} {speed_unit}"

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
                "status": f"Speed: {final_stats['last_speed']} | ETA: {eta_str}"
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

    def run_subprocess(self, command):
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
                errors='replace'
            ) as self.process:

                for output_line in self.process.stdout:
                    output_line = output_line.rstrip()

                    self._parse_final_stats(output_line, final_stats)

                    if 'loading model' in output_line:
                        phase = "Loading Model"
                    elif 'sampling using' in output_line:
                        phase = "Sampling"

                    if "|" in output_line and "/" in output_line:
                        if phase == "Sampling":
                            update_data = self._parse_progress_update(output_line, final_stats)
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


model_state = ModelState()
subprocess_manager = SubprocessManager()


def exe_name():
    """Returns the stable-diffusion executable name"""
    lspci_exists = shutil.which("lspci") is not None
    if not lspci_exists:
        if os.name == "nt":
            return "sd.exe"
        return "./sd"
    else:
        if (shutil.which("sd")):
            return "sd"
        else:
            return "./sd"


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None


def switch_tab_components(
    ckpt_model=None, unet_model=None, ckpt_vae=None, unet_vae=None,
    clip_g=None, clip_l=None, t5xxl=None, pprompt=None, nprompt=None
):
    """Helper function to switch the tab components"""
    return (
        gr.update(value=ckpt_model),
        gr.update(value=unet_model),
        gr.update(value=ckpt_vae),
        gr.update(value=unet_vae),
        gr.update(value=clip_g),
        gr.update(value=clip_l),
        gr.update(value=t5xxl),
        gr.update(
            label=pprompt[0],
            placeholder=pprompt[1]
        ) if pprompt else None,
        gr.update(
            value=nprompt[0],
            visible=nprompt[1]
        ) if nprompt else None
    )


def unet_tab_switch(ckpt_model, ckpt_vae, nprompt):
    """Switches to the UNET tab"""
    model_state.bak_ckpt_tab(ckpt_model, ckpt_vae, nprompt)

    return switch_tab_components(
        ckpt_model=None,
        unet_model=model_state.bak_unet_model,
        ckpt_vae=None,
        unet_vae=model_state.bak_unet_vae,
        clip_g=model_state.bak_clip_g,
        clip_l=model_state.bak_clip_l,
        t5xxl=model_state.bak_t5xxl,
        pprompt=("Prompt", "Prompt"),
        nprompt=(None, False)
    )


def ckpt_tab_switch(unet_model, unet_vae, clip_g, clip_l, t5xxl):
    """Switches to the checkpoint tab"""
    model_state.bak_unet_tab(unet_model, unet_vae, clip_g, clip_l, t5xxl)

    return switch_tab_components(
        ckpt_model=model_state.bak_ckpt_model,
        unet_model=None,
        ckpt_vae=model_state.bak_ckpt_vae,
        unet_vae=None,
        clip_g=None,
        clip_l=None,
        t5xxl=None,
        pprompt=("Positive Prompt", "Positive Prompt"),
        nprompt=(model_state.bak_nprompt, True)
    )


def switch_sizes(height, width):
    return (width, height)

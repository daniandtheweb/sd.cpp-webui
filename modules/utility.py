"""sd.cpp-webui - Utility module"""

import os
import shutil
import subprocess

import gradio as gr

from modules.config import (
    def_sd, def_flux, def_sd_vae, def_flux_vae, def_clip_l, def_t5xxl
)


class ModelState:
    """Class to manage the state of model parameters for the application.

    Attributes:
        bak_sd_model: The backup stable diffusion model.
        bak_flux_model: The backup flux model.
        bak_sd_vae: The backup stable diffusion VAE model.
        bak_flux_vae: The backup flux VAE model.
        bak_clip_l: The backup CLIP model.
        bak_t5xxl: The backup T5-XXL model.
        bak_nprompt: The backup negative prompt.
    """

    def __init__(self):
        """Initializes the ModelState with default values from the
        configuration."""
        self.bak_sd_model = def_sd
        self.bak_flux_model = def_flux
        self.bak_sd_vae = def_sd_vae
        self.bak_flux_vae = def_flux_vae
        self.bak_clip_l = def_clip_l
        self.bak_t5xxl = def_t5xxl
        self.bak_nprompt = None

    def update_sd_tab(self, sd_model, sd_vae, nprompt):
        """Updates the state with values from the Stable-Diffusion tab.

        Args:
            sd_model: The selected stable diffusion model.
            sd_vae: The selected stable diffusion VAE model.
            nprompt: The negative prompt for the Flux tab.
        """
        self.bak_sd_model = sd_model
        self.bak_sd_vae = sd_vae
        self.bak_nprompt = nprompt

    def update_flux_tab(self, flux_model, flux_vae, clip_l, t5xxl):
        """Updates the state with values from the Stable-Diffusion tab.

        Args:
            flux_model: The selected flux model.
            flux_vae: The selected flux VAE model.
            clip_l: The selected CLIP model.
            t5xxl: The selected T5-XXL model.
        """
        self.bak_flux_model = flux_model
        self.bak_flux_vae = flux_vae
        self.bak_clip_l = clip_l
        self.bak_t5xxl = t5xxl


class SubprocessManager:
    """Class to manage subprocess execution and control.

    Attributes:
        process: The currently running subprocess,
                 or None if no subprocess is active.
    """

    def __init__(self):
        """Initializes the SubprocessManager with no active subprocess."""
        self.process = None

    def run_subprocess(self, command):
        """Runs a subprocess with the specified command.

        Args:
            command: A list of command-line arguments for the subprocess.

        This method captures the subprocess's output in real-time and prints
        it.
        If any errors occur during execution, they are also printed after the
        process finishes.
        """
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        ) as self.process:

            # Read the output line by line in real-time
            for output_line in self.process.stdout:
                print(output_line.strip())

            # Wait for the process to finish and capture its errors
            if self.process:
                _, errors = self.process.communicate()
                if errors:
                    print("Errors:", errors)

    def kill_subprocess(self):
        """Terminates the currently running subprocess, if any.

        This method sets the subprocess attribute to None after termination
        and prints a message indicating whether a subprocess was running.
        """
        if self.process is not None:
            self.process.terminate()
            self.process = None
            print("Subprocess terminated.")
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
    return "./sd"


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None


def switch_tab_components(
    sd_model=None, flux_model=None, sd_vae=None, flux_vae=None,
    clip_l=None, t5xxl=None, pprompt=None, nprompt=None
):

    """Helper function to switch the tab components"""
    return (
        gr.update(value=sd_model),
        gr.update(value=flux_model),
        gr.update(value=sd_vae),
        gr.update(value=flux_vae),
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


def flux_tab_switch(sd_model, sd_vae, nprompt):
    """Switches to the Flux tab"""
    model_state.update_sd_tab(sd_model, sd_vae, nprompt)

    return switch_tab_components(
        sd_model=None,
        flux_model=model_state.bak_flux_model,
        sd_vae=None,
        flux_vae=model_state.bak_flux_vae,
        clip_l=model_state.bak_clip_l,
        t5xxl=model_state.bak_t5xxl,
        pprompt=("Prompt", "Prompt"),
        nprompt=(None, False)
    )


def sd_tab_switch(flux_model, flux_vae, clip_l, t5xxl):
    """Switches to the Stable-Diffusion tab"""
    model_state.update_flux_tab(flux_model, flux_vae, clip_l, t5xxl)

    return switch_tab_components(
        sd_model=model_state.bak_sd_model,
        flux_model=None,
        sd_vae=model_state.bak_sd_vae,
        flux_vae=None,
        clip_l=None,
        t5xxl=None,
        pprompt=("Positive Prompt", "Positive Prompt"),
        nprompt=(model_state.bak_nprompt, True)
    )

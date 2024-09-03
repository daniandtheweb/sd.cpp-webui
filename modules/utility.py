"""sd.cpp-webui - Utility module"""

import os
import shutil
import subprocess

import gradio as gr

from modules.config import (
    def_sd, def_flux, def_sd_vae, def_flux_vae, def_clip_l, def_t5xxl
)

global_process = None

bak_sd_model = def_sd
bak_flux_model = def_flux
bak_sd_vae = def_sd_vae
bak_flux_vae = def_flux_vae
bak_clip_l = def_clip_l
bak_t5xxl = def_t5xxl
bak_nprompt = None


def exe_name():
    """Returns the stable-diffusion executable name"""
    lspci_exists = shutil.which("lspci") is not None
    if not lspci_exists:
        if os.name == "nt":
            return "sd.exe"
        return "./sd"
    return "./sd"


def run_subprocess(command):
    """Runs subprocess"""
    global global_process
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True) as global_process:

        # Read the output line by line in real-time
        for output_line in global_process.stdout:
            print(output_line.strip())

        # Wait for the process to finish and capture its errors and print them
        if global_process:
            _, errors = global_process.communicate()
            if errors:
                print("Errors:", errors)


def kill_subprocess():
    """Kills the running subprocess"""
    global global_process
    if global_process is not None:
        global_process.terminate()
        global_process = None
        print("Subprocess terminated.")
    else:
        print("No subprocess running.")


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def get_path(directory, filename):
    """Helper function to construct paths"""
    return os.path.join(directory, filename) if filename else None


def flux_tab_switch(sd_model, sd_vae, nprompt):
    """Switches to the Flux tab"""
    global bak_sd_model
    global bak_sd_vae
    global bak_flux_model
    global bak_flux_vae
    global bak_clip_l
    global bak_t5xxl
    global bak_nprompt
    bak_sd_model = sd_model
    bak_sd_vae = sd_vae
    bak_nprompt = nprompt
    sd_model = gr.update(value=None)
    vae = gr.update(value=None)
    flux_model = gr.update(value=bak_flux_model)
    flux_vae = gr.update(value=bak_flux_vae)
    clip_l = gr.update(value=bak_clip_l)
    t5xxl = gr.update(value=bak_t5xxl)
    pprompt = gr.update(label="Prompt", placeholder="Prompt")
    nprompt = gr.update(value=None, visible=False)
    return (sd_model, flux_model, vae, flux_vae, clip_l, t5xxl, pprompt,
            nprompt)


def sd_tab_switch(flux_model, flux_vae, clip_l, t5xxl):
    """Switches to the Stable-Diffusion tab"""
    global bak_sd_model
    global bak_sd_vae
    global bak_flux_model
    global bak_flux_vae
    global bak_clip_l
    global bak_t5xxl
    global bak_nprompt
    bak_flux_model = flux_model
    bak_flux_vae = flux_vae
    bak_clip_l = clip_l
    bak_t5xxl = t5xxl
    flux_model = gr.update(value=None)
    flux_vae = gr.update(value=None)
    clip_l = gr.update(value=None)
    t5xxl = gr.update(value=None)
    sd_model = gr.update(value=bak_sd_model)
    sd_vae = gr.update(value=bak_sd_vae)
    pprompt = gr.update(label="Positive Prompt", placeholder="Positive Prompt")
    nprompt = gr.update(value=bak_nprompt, visible=True)
    return (sd_model, flux_model, sd_vae, flux_vae, clip_l, t5xxl, pprompt,
            nprompt)

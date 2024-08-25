"""sd.cpp-webui - Utility module"""

import os
import shutil
import subprocess

import gradio as gr

global_process = None


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

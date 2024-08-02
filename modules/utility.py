"""sd.cpp-webui - Utility module"""

import subprocess

global_process = None

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
        global_process=None
        print("Subprocess terminated.")
    else:
        print("No subprocess running.")

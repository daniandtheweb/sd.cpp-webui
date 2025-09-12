import subprocess
import re

def _parse_help_option(option_name: str):
    """
    Parse sd --help output dynamically, streaming stdout,
    so we can get values as soon as they appear.
    Returns a list of values (or empty list if not found).
    """
    pattern = fr"{re.escape(option_name)}.*\{{([^\}}]+)\}}"
    try:
        # Start sd --help as subprocess with stdout piped
        process = subprocess.Popen(
            ["sd", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Read stdout line by line
        help_text = ""
        for line in process.stdout:
            help_text += line
            # Try to match the option immediately
            match = re.search(pattern, help_text)
            if match:
                process.kill()  # stop sd immediately after we found it
                return [v.strip() for v in match.group(1).split(",")]

        # fallback: wait until process exits if not found
        process.wait()
    except Exception:
        pass
    return []  # always return a list

def get_samplers():
    """Return all samplers available in sd."""
    return _parse_help_option("--sampling-method")

def get_schedulers():
    """Return all schedulers available in sd."""
    return _parse_help_option("--scheduler")

def get_previews():
    """Return all preview modes available in sd."""
    return _parse_help_option("--preview")

def get_quants():
    """Return all quantization types available in sd."""
    return _parse_help_option("--type")

def get_prediction():
    """Return all prediction types available in sd."""
    return _parse_help_option("--prediction")

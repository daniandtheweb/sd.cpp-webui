#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

help_print() {
    echo ""
    echo ""
    echo "Usage: ./sdcpp_webui.sh [options]"
    echo ""
    echo "Options:"
    echo "    -h or --help:            Show this help"
    echo "    --listen:                Share sd.cpp-webui on your local network"
    echo "    --autostart:             Open the UI automatically"
    echo "    --darkmode:              Forces the UI to launch in dark mode"
    echo ""
    echo ""
    exit 0
}

for arg in "$@"; do
  case $arg in
    --*'='*) shift; set -- "${arg%%=*}" "${arg#*=}" "$@"; continue;;
    -h|--help) help_print;;
    --listen) ;;       # Passed to python
    --autostart) ;;    # Passed to python
    --darkmode) ;;     # Passed to python
    *) echo "Unknown command parameter: $arg"; exit 1;;
  esac
done

if [ ! -x "sd" ] && ! command -v "sd"; then
  echo ""
  echo ""
  echo "Warning: 'sd' executable not found or doesn't have execute permissions."
  echo "For the command to work place the stable-diffusion.cpp executable in the main sd.cpp-webui folder."
  echo "The executable must be called 'sd'."
  echo ""
  echo ""
fi

if [ -d "venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi
echo "Virtual environment activated succesfully."

if pip freeze | grep -q -F -f requirements.txt; then
    echo "Requirements are satisfied."
else
    echo "Installing requirements..."
    pip install -r requirements.txt --upgrade pip
    echo "Requirements installed."
fi

echo "Starting the WebUI..."
python3 sdcpp_webui.py $@

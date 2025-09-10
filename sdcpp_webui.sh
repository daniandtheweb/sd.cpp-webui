#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

print_help() {
cat << EOF

Usage: ./sdcpp_webui.sh [options]

Options:
    -h or --help:            Show this help
    --listen:                Share sd.cpp-webui on your local network
    --autostart:             Open the UI automatically
    --darkmode:              Forces the UI to launch in dark mode

EOF
exit 0
}

for arg in "$@"; do
  case $arg in
    -h|--help)
      print_help
      ;;
    --listen|--autostart|--darkmode)
      ;;
    *)
      echo "Error: Unknown command parameter: $arg"
      print_help
      ;;
  esac
done

if [ ! -x "sd" ] && ! command -v "sd" > /dev/null; then
  echo "Warning: 'sd' executable not found in this directory or your PATH."
  echo "Please place the stable-diffusion.cpp executable (renamed to 'sd') in this folder."
  exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source "venv/bin/activate"

if pip freeze | grep -q -F -f requirements.txt; then
    echo "Requirements are satisfied."
else
    echo "Installing requirements..."
    pip install -r requirements.txt --upgrade pip
    echo "Requirements installed."
fi

echo "Starting the WebUI..."
python3 sdcpp_webui.py "$@"

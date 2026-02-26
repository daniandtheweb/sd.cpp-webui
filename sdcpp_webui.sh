#!/usr/bin/env bash
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

print_help() {
cat << EOF

Usage: ./sdcpp_webui.sh [options]

Options:
    -h or --help:            Show this help
    --server                 Enable stable-diffusion.cpp's server mode
    --listen:                Share sd.cpp-webui on your local network
    --autostart:             Open the UI automatically
    --darkmode:              Forces the UI to launch in dark mode
    --credentials:           Enable password protection using credentials.json
                             Expected format: {"username1": "password1", "username2": "password2"}
    --allow-insecure-dir:    Allows the usage of external or linked directories based on config.json

EOF
exit 0
}

for arg in "$@"; do
  case $arg in
    -h|--help)
      print_help
      ;;
    --server|--listen|--autostart|--darkmode|--credentials|--allow-insecure-dir)
      ;;
    *)
      printf "Error: Unknown command parameter: %s\n" "$arg" >&2
      print_help
      ;;
  esac
done

if [ ! -x "sd-cli" ] && ! command -v "sd-cli" > /dev/null && \
   [ ! -x "sd" ] && ! command -v "sd" > /dev/null; then
  echo "Warning: Neither 'sd-cli' nor 'sd' executables were found in this directory or your PATH."
  echo "Please place the stable-diffusion.cpp executable (renamed to 'sd-cli' or 'sd') in this folder."
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

if [ $# -eq 0 ]; then
    printf "Starting the WebUI...\n\n"
else
    printf "Starting the WebUI with arguments: %s\n\n" "$*"
fi
python3 sdcpp_webui.py "$@"

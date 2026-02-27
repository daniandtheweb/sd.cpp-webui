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
    --server)
      SERVER_MODE=1
      ;;
    --listen|--autostart|--darkmode|--credentials|--allow-insecure-dir)
      ;;
    *)
      printf "Error: Unknown command parameter: %s\n" "$arg" >&2
      print_help
      ;;
  esac
done

if [ "$SERVER_MODE" -eq 1 ]; then
  if [ ! -x "sd-server" ] && ! command -v "sd-server" > /dev/null; then
    echo "Error: '--server' flag was used, but the 'sd-server' executable was not found or is not executable."
    echo "Please place the 'sd-server' executable in this folder or ensure it is in your PATH."
    exit 1
  fi
elif [ ! -x "sd-cli" ] && ! command -v "sd-cli" > /dev/null && \
   [ ! -x "sd" ] && ! command -v "sd" > /dev/null; then
  echo "Error: Neither 'sd-cli' nor 'sd' executables were found or they're not executable."
  echo "Please place the 'sd-cli' or 'sd' executable in this folder or ensure it is in your PATH."
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

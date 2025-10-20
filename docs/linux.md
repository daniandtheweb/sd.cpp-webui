# Setup instructions for Linux

## Dependencies:

- Python
- Git (Recommended)

## Setup:

1. Clone the repository with Git:
    ```bash
    git clone https://github.com/daniandtheweb/sd.cpp-webui.git
    ```

2. Move inside of `sd.cpp-webui`'s directory:
    ```bash
    cd sd.cpp-webui
    ```

3. Execute the launch script, this will automatically create a virtual environment and install all the necessary Python packages:
    ```bash
    ./sdcpp_webui.sh --autostart
    ```

- To see all available arguments and options, run:
    ```bash
    ./sdcpp_webui.sh --help
    ```

## Updating the WebUI:

1. Inside of `sd.cpp-webui`'s directory run:
    ```bash
    git pull
    ```
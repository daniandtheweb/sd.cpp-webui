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

3. Obtain the `stable-diffusion.cpp` binary:

    - Compile from source:

       1. Obtain the source code:

           ```bash
           git clone https://github.com/leejet/stable-diffusion.cpp.git
           ```

       2. Compile following the instructions from `stable-diffusion.cpp`'s repository.

       3. Copy the compiled `sd` binary to the main `sd.cpp_webui` directory.

    - Download a precompiled binary:

       1. Download a precompiled release from the Releases section of `stable-diffusion.cpp`'s repository.

       2. Copy the `sd` binary to the main `sd.cpp_webui` directory.


4. Execute the launch script, this will automatically create a virtual environment and install all the necessary Python packages:

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

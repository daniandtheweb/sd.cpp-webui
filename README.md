# sd.cpp-webui

A simple interface based on Gradio library for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

This program is essentially a command generator for **stable-diffusion.cpp** with some extra functions.

> [!IMPORTANT]
[2024 Sep 3] There's been a lot of refactoring around the project, if you have any issues try to delete the old `config.json` file, the program will create a new one on the next start.

## Features

- Minimal python dependencies (gradio is the main one, no pytorch required)
- Any **stable-diffusion.cpp** feature
- Gallery
- Metadata reader


## Installation and Running

1. You'll need the following dependencies:
   - git
   - python
2. Clone the repository:
```bash
git clone https://github.com/daniandtheweb/sd.cpp-webui.git
```
3. Clone `stable-diffusion.cpp`:
```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
```
4. Enter `stable-diffusion.cpp`'s folder and follow the instructions to build it.
5. Copy the built file (sd or sd.exe) from `build/bin` to the `sd.cpp-webui` folder.
6. If you're on Linux or MacOS make sure the copied file is executable.
7. Run the launch script: 
   - `sdpp_webui_linux.sh` for Linux
   - `sdcpp_webui_windows.bat` for Windows


If you want to share the WebUI locally on your newtork run the script with `--listen`.


![swappy-20240904-145835](https://github.com/user-attachments/assets/78c52f9e-f6f7-454d-aa77-b3288571fe4e)


## Credits

- stable-diffusion.cpp - [https://github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Gradio - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

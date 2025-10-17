# sd.cpp-webui

A simple Gradio-based interface for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

This program acts primarily as a command generator for **stable-diffusion.cpp**, with some extra features.

## Features

- Minimal python dependencies (Gradio is the main one, no PyTorch required)
- Supports most **stable-diffusion.cpp** features (the missing ones are work in progress)
- Built-in image gallery
- Metadata reader

## **stable-diffusion.cpp**'s supported models

- SD1.x, SD2.x, SD-Turbo, SDXL, SDXL-Turbo, NitroFusion
- SD3, SD3.5, Flux-dev, Flux-schnell, Chroma, Qwen Image
- Flux Kontext, Qwen Image Edit
- Wan2.1, Wan2.2
- TAESD
- PhotoMaker
- ControlNet
- Upscaling models
- LoRAs and embeddings

## Installation and Running

### Dependencies
   - Git
   - Python

### Steps
1. Clone the repository:
```bash
git clone https://github.com/daniandtheweb/sd.cpp-webui.git
```
2. Clone `stable-diffusion.cpp`:
```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
```
3. Follow the instructions in **stable-diffusion.cpp**`s folder to build the project.
4. Copy the built file (`sd` or `sd.exe`) from the `build/bin` directory to the `sd.cpp-webui` folder.
5. On Linux/macOS, ensure the copied file is executable.
6. Run the launch script: 
   - `sdcpp_webui.sh` for Linux/macOS
   - `sdcpp_webui_windows.bat` for Windows


For more information on available launch arguments, run the script with `-h` or `--help`.


![swappy-20240904-145835](https://github.com/user-attachments/assets/78c52f9e-f6f7-454d-aa77-b3288571fe4e)


## Credits

- stable-diffusion.cpp - [https://github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Gradio - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

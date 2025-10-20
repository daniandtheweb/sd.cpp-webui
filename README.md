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

- Python (3.13 on Windows)
- Git (Recommended)

### Setup

Just run [`sdcpp_webui.sh`](https://github.com/daniandtheweb/sd.cpp-webui/blob/master/docs/linux.md) for Linux/MacOS or [`sdcpp_webui_windows.ps1`](https://github.com/daniandtheweb/sd.cpp-webui/blob/master/docs/windows.md)(WIP) for Windows.


<img width="2617" height="1828" alt="sdcpp_webui_screenshot" src="https://github.com/user-attachments/assets/1119195a-6c7c-483d-b475-d0ef6ae96fb0" />


## Contributors

Thank you to all the contributors!

[![Contributors](https://contrib.rocks/image?repo=daniandtheweb/sd.cpp-webui)](https://github.com/daniandtheweb/sd.cpp-webui/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=daniandtheweb/sd.cpp-webui&type=Date)](https://star-history.com/#daniandtheweb/sd.cpp-webui&Date)

## Credits

- stable-diffusion.cpp - [https://github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Gradio - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

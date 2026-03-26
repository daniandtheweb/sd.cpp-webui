# sd.cpp-webui - A Lightweight Gradio WebUI for stable-diffusion.cpp

**sd.cpp-webui** is a simple, lightweight [Gradio](https://github.com/gradio-app/gradio)-based web interface for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

Designed to run local AI text-to-image and video generation models without heavy dependencies, this program acts primarily as a command generator for **stable-diffusion.cpp**, with some extra features for image management.

## Features

- **Lightweight:** Minimal Python dependencies (Gradio is the main requirement, no PyTorch required).
- **Feature support:** Supports most **stable-diffusion.cpp** features (missing features are work in progress).
- **Generation preview:** Live image preview using **stable-diffusion.cpp**'s native preview option.
- **Image management:** Built-in image gallery.
- **Metadata reader:** Built-in metadata reader to extract generation data from images.
- **Secure access:** Optional protected login with user credentials.

## **stable-diffusion.cpp**'s supported models

This WebUI, by using **stable-diffusion.cpp** as its core, supports a large number of image/video generation models, including:

- SD1.x, SD2.x, SD-Turbo, SDXL, SDXL-Turbo, NitroFusion
- SD3, SD3.5, FLUX.1-Krea-dev, FLUX.1-dev, FLUX.1-schnell, FLUX.2-dev, Chroma, Qwen Image, Z-Image-Turbo
- FLUX.1-Kontext-dev, Qwen Image Edit, Qwen Image Edit 2509
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

- Detailed instructions for Linux/MacOS: [`sdcpp_webui.sh`](https://github.com/daniandtheweb/sd.cpp-webui/blob/master/docs/linux.md)
- Detailed instructions for Windows: [`sdcpp_webui_windows.ps1`](https://github.com/daniandtheweb/sd.cpp-webui/blob/master/docs/windows.md)

- Quick Start for experienced users:

  1. Clone the repository:

      ```bash
      git clone https://github.com/daniandtheweb/sd.cpp-webui.git
      ```

  2. Obtain the `stable-diffusion.cpp` binary (sd for Linux/MacOS or sd.exe for Windows) by compiling or downloading it from the releases and place it in the main `sd.cpp-webui` folder.

  3. Run `sdcpp_webui.sh` if you're on Linux/MacOS or `sdcpp_webui_windows.ps1` if you're on Windows.

  4. Access the WebUI with the browser at `http://localhost:7860/`.


<img width="2860" height="1954" alt="Screenshot of sd.cpp-webui, a lightweight Gradio interface for local AI image generation using stable-diffusion.cpp" src="https://github.com/user-attachments/assets/9e6cc19f-55f4-4f76-8202-fa41bbbc4975" />


## Contributors

Thank you to all the contributors!

[![Contributors](https://contrib.rocks/image?repo=daniandtheweb/sd.cpp-webui)](https://github.com/daniandtheweb/sd.cpp-webui/graphs/contributors)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=daniandtheweb/sd.cpp-webui&type=Date)](https://star-history.com/#daniandtheweb/sd.cpp-webui&Date)

## Credits

- stable-diffusion.cpp - [https://github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Gradio - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

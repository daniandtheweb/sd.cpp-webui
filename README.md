# sd.cpp-webui

A simple interface based on Gradio library for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).


## Features

- Minimal python dependencies (gradio is the main one, no pytorch required, only raw CUDA/ROCm is required for GPU inference)
- txt2img and img2img modes
- ControlNet
- Gallery
- Read images metadata
- 4GB video card support (stable-diffusion-xl models quantized q8_0 run with less than 4GB of VRAM)
- Supports gguf, safetensors (ckpt can be converted)


## Installation and Running

1. You'll need the following dependencies:
   - git
   - python
2. Clone the repository:
```bash
git clone https://github.com/DaniAndTheWeb/sd.cpp-webui.git
```
3. Clone `stable-diffusion.cpp`:
```bash
git clone https://github.com/leejet/stable-diffusion.cpp.git
```
4. Enter `stable-diffusion.cpp`'s folder and follow the instructions to build it.
5. Copy the built file (sd or sd.exe) from `build/bin` to the `sd.cpp-webui`'s folder.
6. If you're on Linux or MacOS make sure the copied file is executable.
7. Run the launch script: 
   - `sdpp_webui_linux.sh` for Linux
   - `sdcpp_webui_windows.bat` for Windows


If you want to share the WebUI locally on your newtork run the script with `--listen`.


![sd_cpp-webui](https://github.com/DaniAndTheWeb/sd.cpp-webui/assets/57776841/0fa8e459-61db-4993-bead-cfc9032532b3)


## Credits

- stable-diffusion.cpp - [https://github.com/leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
- Gradio - [https://github.com/gradio-app/gradio](https://github.com/gradio-app/gradio)

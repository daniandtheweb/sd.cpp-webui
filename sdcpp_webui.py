import os
import gradio as gr
import subprocess

def get_models():
    models_folder = "models/Stable-Diffusion"
    if os.path.isdir(models_folder):
        return [model for model in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, model))]
    else:
        print(f"The {models_folder} folder does not exist.")
        return []

def get_hf_models():
    models_folder = "models/Stable-Diffusion"
    if os.path.isdir(models_folder):
        return [model for model in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, model)) and (model.endswith(".safetensors") or model.endswith(".ckpt"))]
    else:
        print(f"The {models_folder} folder does not exist.")
        return []

def get_vaes():
    models_folder = "models/VAE"
    if os.path.isdir(models_folder):
        return [model for model in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, model))]
    else:
        print(f"The {models_folder} folder does not exist.")
        return []

def get_next_txt2img():
    txt2img_out = "outputs/txt2img"
    files = os.listdir(txt2img_out)
    png_files = [file for file in files if file.endswith('.png')]

    if not png_files:
        return '1.png'

    highest_number = max([int(file.split('.')[0]) for file in png_files])
    next_number = highest_number + 1
    return f"{next_number}.png"

def txt2img(model, vae, ppromt, nprompt, sampling, steps, width, height, cfg, seed, threads, output):
    fmodel = f'models/Stable-Diffusion/{model}'
    if vae:    
        fvae = f'models/VAE/{vae}'
    fpprompt = f'"{ppromt}"'
    if nprompt:
        fnprompt = f'"{nprompt}"'    
    fsampling = f'{sampling}'
    fsteps = str(steps)
    fwidth = str(width)
    fheight = str(height)
    fcfg = str(cfg)
    fseed = str(seed)
    fthreads = str(threads)

    if output is None or '""':
        foutput = "outputs/txt2img/" + get_next_txt2img()
    else:
        foutput = f'"outputs/txt2img/{output}.png"'

    command = ['./sd', '-M', 'txt2img', '-m', fmodel, '-p', fpprompt, '--sampling-method', fsampling, '--steps', fsteps, '-W', fwidth, '-H', fheight, '--cfg-scale', fcfg, '-s', fseed, '-t', fthreads, '-o', foutput]

    if 'fvae' in locals():
        command.extend(['--vae', fvae])
    if 'fnprompt' in locals():
        command.extend(['-n', fnprompt])

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and capture its output and errors
    output, errors = process.communicate()

    # Print the output and errors (if any)
    print("Output:", output.decode())
    print("Errors:", errors.decode())
    return foutput

def convert(og_model, type, gguf_model):
    fog_model =  f'models/Stable-Diffusion/{og_model}'
    ftype = f'{type}'
    if gguf_model is None or '""':
        fgguf_model = f'"models/Stable-Diffusion/{og_model}.{type}.gguf"'
    else:
        fgguf_model = f'"models/Stable-Diffusion/{gguf_model}"'

    command=['./sd', '-M', 'convert', '-m', fog_model, '-o', fgguf_model, '--type', ftype]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to finish and capture its output and errors
    output, errors = process.communicate()

    # Print the output and errors (if any)
    print("Output:", output.decode())
    print("Errors:", errors.decode())
    return "Process Complete."

def greet(name):
    return "Hello " + name + "!"

txt2img = gr.Interface(
    fn=txt2img,
    inputs=[
        gr.Dropdown(label="Model", choices=get_models()),
        gr.Dropdown(label="VAE", choices=get_vaes()),
        gr.Textbox(label="Positive Prompt"),
        gr.Textbox(label="Negative Prompt"),
        gr.Dropdown(label="Sampling method", choices=["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "lcm"], value="euler_a"),
        gr.Slider(label="Steps", minimum=1, maximum=999, value=20),
        gr.Slider(label="Width", minimum=1, maximum=8192, value=512),
        gr.Slider(label="Height", minimum=1, maximum=8192, value=512),
        gr.Slider(label="CFG Scale", minimum=1, maximum=30, value=7.0),
        gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1),
        gr.Number(label="Threads", minimum=0, maximum=str(os.cpu_count()), value=0),
        gr.Textbox(label="Output Name (optional)", value="")],
    outputs="image",
    title="Text to Image",
    allow_flagging="never",
)

img2img = gr.Interface(
fn=greet,
inputs="textbox",
outputs="textbox",
title="Image to Image (WIP)",
allow_flagging="never",
)

convert = gr.Interface(
    fn=convert,
    inputs=[
        gr.Dropdown(label="Original Model", choices=get_hf_models()),
        gr.Dropdown(label="Type", choices=["f32", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"], value="f32"),
        gr.Textbox(label="Output Name (oprional, must end with .gguf)", value=""),],
    outputs=[
        "text",],
    title="Convert and Quantize (WIP)",
    allow_flagging="never",
)

sdcpp = gr.TabbedInterface(
    [txt2img, img2img, convert],
    ["txt2img", "img2img", "Checkpoint Converter"],
    title="SDCPP",
)

sdcpp.launch()

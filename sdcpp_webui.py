import os
import gradio as gr
import subprocess


def get_models(models_folder):
    if os.path.isdir(models_folder):
        return [model for model in os.listdir(models_folder)
                if os.path.isfile(os.path.join(models_folder, model))]
    else:
        print(f"The {models_folder} folder does not exist.")
        return []


def get_hf_models():
    models_folder = "models/Stable-Diffusion"
    if os.path.isdir(models_folder):
        return [model for model in os.listdir(models_folder)
                if os.path.isfile(os.path.join(models_folder, model)) and
                (model.endswith(".safetensors") or model.endswith(".ckpt"))]
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


def txt2img(model, vae, taesd, controlnet, control_img, control_strength,
            ppromt, nprompt, sampling, steps, schedule, width, height,
            batch_count, cfg, seed, clip_skip, threads, vae_tiling,
            cont_net_cpu, rng, output, verbose):
    fmodel = f'models/Stable-Diffusion/{model}'
    if vae:
        fvae = f'models/VAE/{vae}'
    fembed = f'models/Embeddings/'
    flora = f'models/Lora/'
    if taesd:
        ftaesd = f'models/TAESD/{taesd}'
    if controlnet:
        fcontrolnet = f'models/ControlNet/{controlnet}'
    if control_img:
        fcontrol_img = f'{control_img}'
    if control_strength:
        fcontrol_strength = str(control_strength)
    fpprompt = f'"{ppromt}"'
    if nprompt:
        fnprompt = f'"{nprompt}"'
    fsampling = f'{sampling}'
    fsteps = str(steps)
    fschedule = f'{schedule}'
    fwidth = str(width)
    fheight = str(height)
    fbatch_count = str(batch_count)
    fcfg = str(cfg)
    fseed = str(seed)
    fclip_skip = str(clip_skip + 1)
    fthreads = str(threads)
    if vae_tiling:
        fvae_tiling = vae_tiling
    if cont_net_cpu:
        fcont_net_cpu = cont_net_cpu
    frng = f'{rng}'

    if output is None or '""':
        foutput = "outputs/txt2img/" + get_next_txt2img()
    else:
        foutput = f'"outputs/txt2img/{output}.png"'

    if verbose:
        fverbose = verbose

    command = ['./sd', '-M', 'txt2img', '-m', fmodel, '-p', fpprompt,
               '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--cfg-scale', fcfg, '-s', fseed, '--clip-skip',
               fclip_skip, '-t', fthreads, '--rng', frng, '-o', foutput]

    if 'fvae' in locals():
        command.extend(['--vae', fvae])
    if 'fembed' in locals():
        command.extend(['--embd-dir', fembed])
    if 'flora' in locals():
        command.extend(['--lora-model-dir', flora])
    if 'ftaesd' in locals():
        command.extend(['--taesd', ftaesd])
    if 'fcontrolnet' in locals():
        command.extend(['--control-net', fcontrolnet])
    if 'fcontrol_img' in locals():
        command.extend(['--control_image', fcontrol_img])
    if 'fcontrol_strength' in locals():
        command.extend(['--control-strength', fcontrol_strength])
    if 'fnprompt' in locals():
        command.extend(['-n', fnprompt])
    if 'fvae_tiling' in locals():
        command.extend(['--vae-tiling'])
    if 'fcont_net_cpu' in locals():
        command.extend(['--cont_net_cpu'])
    if 'fverbose' in locals():
        command.extend(['-v'])

    print(command)
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)

    # Read the output line by line in real-time
    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print(output_line.strip())

    # Wait for the process to finish and capture its errors
    _, errors = process.communicate()

    # Print any remaining errors (if any)
    if errors:
        print("Errors:", errors)

    img_final = [foutput]
    return img_final


def img2img(model, vae, taesd, img_inp, controlnet, control_img,
            control_strength, ppromt, nprompt, sampling, steps, schedule,
            width, height, batch_count, strenght, cfg, seed, clip_skip,
            threads, vae_tiling, cont_net_cpu, rng, output, verbose):
    fmodel = f'models/Stable-Diffusion/{model}'
    if vae:
        fvae = f'models/VAE/{vae}'
    fembed = f'models/Embeddings/'
    flora = f'models/Lora/'
    if taesd:
        ftaesd = f'models/TAESD/{taesd}'
    fimg_inp = f'{img_inp}'
    if controlnet:
        fcontrolnet = f'models/ControlNet/{controlnet}'
    if control_img:
        fcontrol_img = f'{control_img}'
    if control_strength:
        fcontrol_strength = str(control_strength)
    fpprompt = f'"{ppromt}"'
    if nprompt:
        fnprompt = f'"{nprompt}"'
    fsampling = f'{sampling}'
    fsteps = str(steps)
    fschedule = f'{schedule}'
    fwidth = str(width)
    fheight = str(height)
    fbatch_count = str(batch_count)
    fstrenght = str(strenght)
    fcfg = str(cfg)
    fseed = str(seed)
    fclip_skip = str(clip_skip + 1)
    fthreads = str(threads)
    if vae_tiling:
        fvae_tiling = vae_tiling
    if cont_net_cpu:
        fcont_net_cpu = cont_net_cpu
    frng = f'{rng}'

    if output is None or '""':
        foutput = "outputs/txt2img/" + get_next_txt2img()
    else:
        foutput = f'"outputs/txt2img/{output}.png"'

    if verbose:
        fverbose = verbose

    command = ['./sd', '-M', 'img2img', '-m', fmodel, '-i', fimg_inp, '-p',
               fpprompt, '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--strength', fstrenght, '--cfg-scale', fcfg,
               '-s', fseed, '--clip-skip', fclip_skip, '-t', fthreads, '--rng',
               frng, '-o', foutput]

    if 'fvae' in locals():
        command.extend(['--vae', fvae])
    if 'fembed' in locals():
        command.extend(['--embd-dir', fembed])
    if 'flora' in locals():
        command.extend(['--lora-model-dir', flora])
    if 'ftaesd' in locals():
        command.extend(['--taesd', ftaesd])
    if 'fcontrolnet' in locals():
        command.extend(['--control-net', fcontrolnet])
    if 'fcontrol_img' in locals():
        command.extend(['--control_image', fcontrol_img])
    if 'fcontrol_strength' in locals():
        command.extend(['--control-strength', fcontrol_strength])
    if 'fnprompt' in locals():
        command.extend(['-n', fnprompt])
    if 'fvae_tiling' in locals():
        command.extend(['--vae-tiling'])
    if 'fcont_net_cpu' in locals():
        command.extend(['--cont_net_cpu'])
    if 'fverbose' in locals():
        command.extend(['-v'])

    print(command)
    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)

    # Read the output line by line in real-time
    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print(output_line.strip())

    # Wait for the process to finish and capture its errors
    _, errors = process.communicate()

    # Print any remaining errors (if any)
    if errors:
        print("Errors:", errors)
    img_final = [foutput]
    return foutput


def convert(og_model, type, gguf_model, verbose):
    fog_model = f'models/Stable-Diffusion/{og_model}'
    ftype = f'{type}'
    if gguf_model == '':
        model_name, ext = os.path.splitext(og_model)
        fgguf_model = f'"models/Stable-Diffusion/{model_name}.{type}.gguf"'
    else:
        fgguf_model = f'"models/Stable-Diffusion/{gguf_model}"'
    if verbose:
        fverbose = verbose

    command = ['./sd', '-M', 'convert', '-m', fog_model,
               '-o', fgguf_model, '--type', ftype]

    if 'fverbose' in locals():
        command.extend(['-v'])

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)

    # Read the output line by line in real-time
    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print(output_line.strip())

    # Wait for the process to finish and capture its errors
    _, errors = process.communicate()

    # Print any remaining errors (if any)
    if errors:
        print("Errors:", errors)
    result = "Process completed."
    return result


with gr.Blocks() as txt2img_block:
    txt2img_title = gr.Markdown("# Text to Image"),
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(label="Model",
                                choices=get_models("models/Stable-Diffusion"))
        with gr.Column():
            vae = gr.Dropdown(label="VAE", choices=get_models("models/VAE"))
            clear_model = gr.ClearButton(vae, size="sm")
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            taesd = gr.Dropdown(label="TAESD",
                                choices=get_models("models/TAESD"))
            clear_model = gr.ClearButton(taesd, size="sm")
    with gr.Row():
        with gr.Column(scale=3):
            pprompt = gr.Textbox(placeholder="Positive prompt",
                                 label="Positive Prompt", lines=3)
            nprompt = gr.Textbox(placeholder="Negative prompt",
                                 label="Negative Prompt", lines=3)
        with gr.Column(scale=1):
            gen_btn = gr.Button(value="Generate")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=["euler", "euler_a", "heun",
                                                    "dpm2", "dpm2++2s_a",
                                                    "dpm++2m", "dpm++2mv2",
                                                    "lcm"], value="euler_a")
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=20, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule", choices=["discrete",
                                                                  "karras"],
                                       value="discrete")
            with gr.Row():
                with gr.Column(scale=2):
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=512, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=512, step=1)
                with gr.Column(scale=1):
                    batch_count = gr.Slider(label="Batch count", minimum=1,
                                            maximum=99, value=1, step=1)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            value=7.0, step=0.1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)
            with gr.Accordion(label="ControlNet", open=False):
                controlnet = gr.Dropdown(label="ControlNet",
                                         choices=get_models("models/ControlNet"
                                                            ))
                clear_model = gr.ClearButton(controlnet, size="sm")
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                with gr.Row():
                    with gr.Column(scale=1):
                        vae_tiling = gr.Checkbox(label="Vae Tiling")
                    with gr.Column(scale=1):
                        cont_net_cpu = gr.Checkbox(label="ControlNet on CPU")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                output = gr.Textbox(label="Output Name",
                                    placeholder="Optional")
                verbose = gr.Checkbox(label="Verbose")
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")
            gen_btn.click(txt2img, inputs=[model, vae, taesd, controlnet,
                                           control_img, control_strength,
                                           pprompt, nprompt, sampling, steps,
                                           schedule, width, height,
                                           batch_count, cfg, seed, clip_skip,
                                           threads, vae_tiling, cont_net_cpu,
                                           rng, output, verbose],
                          outputs=[img_final])

with gr.Blocks()as img2img_block:
    img2img_title = gr.Markdown("# Image to Image")
    with gr.Row():
        with gr.Column():
            model = gr.Dropdown(label="Model",
                                choices=get_models("models/Stable-Diffusion"))
        with gr.Column():
            vae = gr.Dropdown(label="VAE", choices=get_models("models/VAE"))
            clear_model = gr.ClearButton(vae, size="sm")
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            taesd = gr.Dropdown(label="TAESD",
                                choices=get_models("models/TAESD"))
            clear_model = gr.ClearButton(taesd, size="sm")
    with gr.Row():
        with gr.Column(scale=3):
            pprompt = gr.Textbox(placeholder="Positive prompt",
                                 label="Positive Prompt", lines=3)
            nprompt = gr.Textbox(placeholder="Negative prompt",
                                 label="Negative Prompt", lines=3)
        with gr.Column(scale=1):
            img_inp = gr.Image(sources="upload", type="filepath")
            gen_btn = gr.Button(value="Generate")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=["euler", "euler_a", "heun",
                                                    "dpm2", "dpm2++2s_a",
                                                    "dpm++2m", "dpm++2mv2",
                                                    "lcm"], value="euler_a")
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=20, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule",
                                       choices=["discrete", "karras"],
                                       value="discrete")
            with gr.Row():
                with gr.Column(scale=2):
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=512, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=512, step=1)
                with gr.Column(scale=1):
                    batch_count = gr.Slider(label="Batch count", minimum=1,
                                            maximum=99, step=1, value=1)
            strenght = gr.Slider(label="Noise strenght", minimum=0, maximum=1,
                                 step=0.01, value=0.75)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            step=0.1, value=7.0)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)
            with gr.Accordion(label="ControlNet", open=False):
                controlnet = gr.Dropdown(label="ControlNet",
                                         choices=get_models("models/ControlNet"
                                                            ))
                clear_model = gr.ClearButton(controlnet, size="sm")
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                with gr.Row():
                    with gr.Column(scale=1):
                        vae_tiling = gr.Checkbox(label="Vae Tiling")
                    with gr.Column(scale=1):
                        cont_net_cpu = gr.Checkbox(label="ControlNet on CPU")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                output = gr.Textbox(label="Output Name (optional)", value="")
                verbose = gr.Checkbox(label="Verbose")
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")
            gen_btn.click(img2img, inputs=[model, vae, taesd, img_inp,
                                           controlnet, control_img,
                                           control_strength, pprompt,
                                           nprompt, sampling, steps, schedule,
                                           width, height, batch_count,
                                           strenght, cfg, seed, clip_skip,
                                           threads, vae_tiling, cont_net_cpu,
                                           rng, output, verbose],
                          outputs=[img_final])

with gr.Blocks() as convert_block:
    convert_title = gr.Markdown("# Convert and Quantize")
    with gr.Row():
        with gr.Column(scale=1):
            og_model = gr.Dropdown(label="Original Model",
                                   choices=get_hf_models())
            type = gr.Dropdown(label="Type",
                               choices=["f32", "f16", "q8_0", "q5_1", "q5_0",
                                        "q4_1", "q4_0"], value="f32")
            verbose = gr.Checkbox(label="Verbose")
            gguf_model = gr.Textbox(label="Output Name (optional, must end "
                                    "with .gguf)", value="")
            convert_btn = gr.Button(value="Convert")
        with gr.Column(scale=1):
            result = gr.Textbox(interactive=False, value="")
    convert_btn.click(convert, inputs=[og_model, type, gguf_model, verbose],
                      outputs=[result])

sdcpp = gr.TabbedInterface(
    [txt2img_block, img2img_block, convert_block],
    ["txt2img", "img2img", "Checkpoint Converter"],
    title="sd.cpp-webui",
    theme=gr.themes.Soft(),
)

sdcpp.launch()

import os
import argparse
import subprocess
import platform
import gradio as gr


current_dir = os.getcwd()
model_dir = "models/Stable-Diffusion"
vae_dir = "models/VAE"
emb_dir = "models/Embeddings"
lora_dir = "models/Lora"
taesd_dir = "models/TAESD"
cnnet_dir = "models/ControlNet"
reload_symbol = '\U0001f504'


if not os.system("which lspci > /dev/null") == 0:
    if os.name == "nt":
        sd = "sd.exe"
    elif os.name == "posix":
        sd = "./sd"
else:
    sd = "./sd"


def get_models(model_dir):
    fmodels_dir = os.path.join(current_dir, model_dir)
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(os.path.join(fmodels_dir, model)) and
                (model.endswith(".gguf") or model.endswith(".safetensors"))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def get_hf_models():
    fmodels_dir = os.path.join(current_dir, "models/Stable-Diffusion")
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(os.path.join(fmodels_dir, model)) and
                (model.endswith(".safetensors") or model.endswith(".ckpt"))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def reload_models(model_dir):
    refreshed_models = gr.update(choices=get_models(model_dir))
    return refreshed_models


def get_next_txt2img():
    ftxt2img_out = os.path.join(current_dir, "outputs/txt2img")
    files = os.listdir(ftxt2img_out)
    png_files = [file for file in files if file.endswith('.png')]

    if not png_files:
        return "1.png"

    highest_number = max([int(file.split('.')[0]) for file in png_files])
    next_number = highest_number + 1
    return f"{next_number}.png"


def txt2img(model, vae, taesd, cnnet, control_img, control_strength,
            ppromt, nprompt, sampling, steps, schedule, width, height,
            batch_count, cfg, seed, clip_skip, threads, vae_tiling,
            cont_net_cpu, rng, output, verbose):
    if model != "None":
        fmodel = os.path.join(current_dir, f'models/Stable-Diffusion/{model}')
    if vae != "None":
        fvae = os.path.join(current_dir, f'models/VAE/{vae}')
    fembed = os.path.join(current_dir, f'models/Embeddings')
    flora = os.path.join(current_dir, f'models/Lora')
    if taesd:
        ftaesd = os.path.join(current_dir, f'models/TAESD/{taesd}')
    if cnnet:
        fcnnet = os.path.join(current_dir,
                              f'models/ControlNet/{cnnet}')
        fcontrol_img = f'{control_img}'
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
        foutput = os.path.join(current_dir, "outputs/txt2img/" +
                               get_next_txt2img())
    else:
        foutput = os.path.join(current_dir, f'"outputs/txt2img/{output}.png"')
    if verbose:
        fverbose = verbose

    command = [sd, '-M', 'txt2img', '-m', fmodel, '-p', fpprompt,
               '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--cfg-scale', fcfg, '-s', fseed, '--clip-skip',
               fclip_skip, '--embd-dir', fembed, '--lora-model-dir', flora,
               '-t', fthreads, '--rng', frng, '-o', foutput]

    if 'fvae' in locals():
        command.extend(['--vae', fvae])
    if 'ftaesd' in locals():
        command.extend(['--taesd', ftaesd])
    if 'fcnnet' in locals():
        command.extend(['--control-net', fcnnet])
        command.extend(['--control_image', fcontrol_img])
        command.extend(['--control-strength', fcontrol_strength])
    if 'fnprompt' in locals():
        command.extend(['-n', fnprompt])
    if 'fvae_tiling' in locals():
        command.extend(['--vae-tiling'])
    if 'fcont_net_cpu' in locals():
        command.extend(['--cont_net_cpu'])
    if 'fverbose' in locals():
        command.extend(['-v'])

    fcommand = ' '.join(str(arg) for arg in command)

    print(fcommand)
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


def img2img(model, vae, taesd, img_inp, cnnet, control_img,
            control_strength, ppromt, nprompt, sampling, steps, schedule,
            width, height, batch_count, strenght, cfg, seed, clip_skip,
            threads, vae_tiling, cont_net_cpu, rng, output, verbose):
    fmodel = os.path.join(current_dir, f'models/Stable-Diffusion/{model}')
    if vae:
        fvae = os.path.join(current_dir, f'models/VAE/{vae}')
    fembed = os.path.join(current_dir, f'models/Embeddings/')
    flora = os.path.join(current_dir, f'models/Lora/')
    if taesd:
        ftaesd = os.path.join(current_dir, f'models/TAESD/{taesd}')
    fimg_inp = f'{img_inp}'
    if cnnet:
        fcnnet = os.path.join(current_dir,
                              f'models/ControlNet/{cnnet}')
        fcontrol_img = f'{control_img}'
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
        foutput = os.path.join(current_dir, "outputs/txt2img/" +
                               get_next_txt2img())
    else:
        foutput = os.path.join(current_dir, f'"outputs/txt2img/{output}.png"')
    if verbose:
        fverbose = verbose

    command = [sd, '-M', 'img2img', '-m', fmodel, '-i', fimg_inp, '-p',
               fpprompt, '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--strength', fstrenght, '--cfg-scale', fcfg,
               '-s', fseed, '--clip-skip', fclip_skip, '--embd-dir', fembed,
               '--lora-model-dir', flora, '-t', fthreads, '--rng', frng, '-o',
               foutput]

    if 'fvae' in locals():
        command.extend(['--vae', fvae])
    if 'ftaesd' in locals():
        command.extend(['--taesd', ftaesd])
    if 'fcnnet' in locals():
        command.extend(['--control-net', fcnnet])
        command.extend(['--control_image', fcontrol_img])
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
    model_dir = os.path.join(current_dir, "models/Stable-Diffusion/")
    fog_model = os.path.join(model_dir, og_model)
    ftype = f'{type}'
    if gguf_model == '':
        model_name, ext = os.path.splitext(og_model)
        fgguf_model = os.path.join(model_dir, f"{model_name}.{type}.gguf")
    else:
        fgguf_model = os.path.join(model_dir, gguf_model)
    if verbose:
        fverbose = verbose

    command = [sd, '-M', 'convert', '-m', fog_model,
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
    model_dir_txt = gr.Textbox(value=model_dir, render=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, render=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, render=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, render=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, render=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, render=False)

    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7)
        rl_model = gr.Button(value=reload_symbol, scale=1)
        rl_model.click(reload_models, inputs=[model_dir_txt], outputs=[model])
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7)
        with gr.Column(scale=1):
            rl_vae = gr.Button(value=reload_symbol)
            rl_vae.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
            clear_vae = gr.ClearButton(vae)
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    rl_taesd = gr.Button(value=reload_symbol, scale=1)
                    rl_taesd.click(reload_models, inputs=[taesd_dir_txt],
                                   outputs=[taesd])
                    clear_taesd = gr.ClearButton(taesd, scale=1)
    with gr.Row():
        with gr.Column(scale=3):
            pprompt = gr.Textbox(placeholder="Positive prompt",
                                 label="Positive Prompt", lines=3,
                                 show_copy_button=True)
            nprompt = gr.Textbox(placeholder="Negative prompt",
                                 label="Negative Prompt", lines=3,
                                 show_copy_button=True)
        with gr.Column(scale=1):
            gen_btn = gr.Button(value="Generate", size="lg")

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
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=512, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=512, step=1)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, value=1, step=1)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            value=7.0, step=0.1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                rl_connet = gr.Button(value=reload_symbol)
                rl_connet.click(reload_models, inputs=[cnnet_dir_txt],
                                outputs=[cnnet])
                clear_connet = gr.ClearButton(cnnet)
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
            gen_btn.click(txt2img, inputs=[model, vae, taesd, cnnet,
                                           control_img, control_strength,
                                           pprompt, nprompt, sampling, steps,
                                           schedule, width, height,
                                           batch_count, cfg, seed, clip_skip,
                                           threads, vae_tiling, cont_net_cpu,
                                           rng, output, verbose],
                          outputs=[img_final])


with gr.Blocks()as img2img_block:
    img2img_title = gr.Markdown("# Image to Image")
    model_dir_txt = gr.Textbox(value=model_dir, render=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, render=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, render=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, render=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, render=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, render=False)

    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7)
        rl_model = gr.Button(value=reload_symbol, scale=1)
        rl_model.click(reload_models, inputs=[model_dir_txt], outputs=[model])
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7)
        with gr.Column(scale=1):
            rl_vae = gr.Button(value=reload_symbol)
            rl_vae.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
            clear_vae = gr.ClearButton(vae)
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    rl_taesd = gr.Button(value=reload_symbol, scale=1)
                    rl_taesd.click(reload_models, inputs=[taesd_dir_txt],
                                   outputs=[taesd])
                    clear_taesd = gr.ClearButton(taesd, scale=1)
    with gr.Row():
        with gr.Column(scale=3):
            pprompt = gr.Textbox(placeholder="Positive prompt",
                                 label="Positive Prompt", lines=3,
                                 show_copy_button=True)
            nprompt = gr.Textbox(placeholder="Negative prompt",
                                 label="Negative Prompt", lines=3,
                                 show_copy_button=True)
        with gr.Column(scale=1):
            gen_btn = gr.Button(value="Generate")
            img_inp = gr.Image(sources="upload", type="filepath")
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
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=512, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=512, step=1)
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
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                rl_connet = gr.Button(value=reload_symbol)
                rl_connet.click(reload_models, inputs=[cnnet_dir_txt],
                                outputs=[cnnet])
                clear_connet = gr.ClearButton(cnnet)
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
                                           cnnet, control_img,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process optional arguments')
    parser.add_argument('--listen', action='store_true',
                        help='Listen on 0.0.0.0')
    args = parser.parse_args()
    if args.listen:
        sdcpp.launch(server_name="0.0.0.0")
    else:
        sdcpp.launch()

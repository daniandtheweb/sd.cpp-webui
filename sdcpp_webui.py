import os
import argparse
import subprocess
import platform
import gradio as gr
from PIL import Image
import numpy as np
import json
import piexif
import piexif.helper


current_dir = os.getcwd()
json_path = 'config.json'
reload_symbol = '\U0001f504'
page_num = 0
ctrl = 0


with open(json_path, 'r') as json_file:
    data = json.load(json_file)


model_dir = data['model_dir']
vae_dir = data['vae_dir']
emb_dir = data['emb_dir']
lora_dir = data['lora_dir']
taesd_dir = data['taesd_dir']
cnnet_dir = data['cnnet_dir']
txt2img_dir = data['txt2img_dir']
img2img_dir = data['img2img_dir']


if 'def_model' in data:
    def_model = data['def_model']
else:
    def_model = None
if 'def_vae' in data:
    def_vae = data['def_vae']
else:
    def_vae = None
def_sampling = data['def_sampling']
def_steps = data['def_steps']
def_schedule = data['def_schedule']
def_width = data['def_width']
def_height = data['def_height']


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
                (model.endswith((".gguf", ".safetensors", ".pth")))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def reload_models(model_dir):
    refreshed_models = gr.update(choices=get_models(model_dir))
    return refreshed_models


def get_hf_models():
    fmodels_dir = os.path.join(current_dir, model_dir)
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(os.path.join(fmodels_dir, model)) and
                (model.endswith((".safetensors", ".ckpt", ".pth", ".gguf")))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def reload_hf_models():
    refreshed_models = gr.update(choices=get_hf_models())
    return refreshed_models


def reload_gallery(ctrl_inp=None, fpage_num=1, subctrl=0):
    global ctrl
    global page_num
    if ctrl_inp is not None:
        ctrl = int(ctrl_inp)
    imgs = []
    if ctrl == 0:
        img_dir = txt2img_dir
    elif ctrl == 1:
        img_dir = img2img_dir
    files = os.listdir(img_dir)
    image_files = [file for file in files if file.endswith(('.jpg', '.png'))]
    image_files.sort()
    start_index = (fpage_num * 16) - 16
    end_index = min(start_index + 16, len(image_files))
    for file_name in image_files[start_index:end_index]:
        image_path = os.path.join(img_dir, file_name)
        image = Image.open(image_path)
        imgs.append(image)
    page_num = fpage_num
    if subctrl == 0:
        return imgs, page_num
    else:
        return imgs


def goto_gallery(fpage_num=1):
    global ctrl
    global page_num
    imgs = []
    if ctrl == 0:
        img_dir = txt2img_dir
    elif ctrl == 1:
        img_dir = img2img_dir
    files = os.listdir(img_dir)
    total_imgs = len([file for file in files if
                      file.endswith(('.png', '.jpg'))])
    total_pages = (total_imgs + 15) // 16
    if fpage_num is None:
        fpage_num = 1
    if fpage_num > total_pages:
        fpage_num = total_pages
    files = os.listdir(img_dir)
    image_files = [file for file in files if file.endswith(('.jpg', '.png'))]
    image_files.sort()
    start_index = (fpage_num * 16) - 16
    end_index = min(start_index + 16, len(image_files))
    for file_name in image_files[start_index:end_index]:
        image_path = os.path.join(img_dir, file_name)
        image = Image.open(image_path)
        imgs.append(image)
    page_num = fpage_num
    return imgs, page_num


def next_page():
    global ctrl
    global page_num
    ctrl_inp = ctrl
    subctrl = 1
    imgs = []
    next_page_num = page_num + 1
    if ctrl == 0:
        files = os.listdir(txt2img_dir)
    elif ctrl == 1:
        files = os.listdir(img2img_dir)
    total_imgs = len([file for file in files if
                      file.endswith(('.png', '.jpg'))])
    total_pages = (total_imgs + 15) // 16
    if next_page_num > total_pages:
        page_num = 1
        imgs = reload_gallery(ctrl_inp, page_num, subctrl)
    else:
        page_num = next_page_num
        imgs = reload_gallery(ctrl_inp, next_page_num, subctrl)
    return imgs, page_num


def prev_page():
    global ctrl
    global page_num
    ctrl_inp = ctrl
    subctrl = 1
    imgs = []
    prev_page_num = page_num - 1
    if ctrl == 0:
        files = os.listdir(txt2img_dir)
    elif ctrl == 1:
        files = os.listdir(img2img_dir)
    total_imgs = len([file for file in files if
                      file.endswith(('.png', '.jpg'))])
    total_pages = (total_imgs + 15) // 16
    if prev_page_num < 1:
        page_num = total_pages
        imgs = reload_gallery(ctrl_inp, total_pages, subctrl)
    else:
        page_num = prev_page_num
        imgs = reload_gallery(ctrl_inp, prev_page_num, subctrl)
    return imgs, page_num


def last_page():
    global ctrl
    global page_num
    ctrl_inp = ctrl
    subctrl = 1
    if ctrl == 0:
        files = os.listdir(txt2img_dir)
    elif ctrl == 1:
        files = os.listdir(img2img_dir)
    total_imgs = len([file for file in files if
                      file.endswith(('.png', '.jpg'))])
    total_pages = (total_imgs + 15) // 16
    imgs = reload_gallery(ctrl_inp, total_pages, subctrl)
    page_num = total_pages
    return imgs, page_num


def img_info(sel_img: gr.SelectData):
    global ctrl
    global page_num
    img_index = (page_num * 16) - 16 + sel_img.index
    if ctrl == 0:
        img_dir = txt2img_dir
    elif ctrl == 1:
        img_dir = img2img_dir
    file_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg')):
                file_paths.append(os.path.join(root, file))
    file_paths.sort()
    try:
        img_path = file_paths[img_index]
    except IndexError:
        print("Image index is out of range.")
        return
    if img_path.endswith(('.jpg', '.jpeg')):
        exif_data = piexif.load(img_path)
        user_comment = piexif.helper.UserComment.load(
                       exif_data["Exif"][piexif.ExifIFD.UserComment])
        metadata = f"JPG: Exif\nPositive prompt: {user_comment}"
        return metadata
    elif img_path.endswith('.png'):
        with open(img_path, 'rb') as f:
            header = f.read(8)
            while True:
                length_chunk = f.read(4)
                if not length_chunk:
                    break
                length = int.from_bytes(length_chunk, byteorder='big')
                chunk_type = f.read(4).decode('utf-8')
                data = f.read(length)
                crc = f.read(4)
                if chunk_type == 'tEXt':
                    keyword, value = data.split(b'\x00', 1)
                    tEXt = f"{value.decode('utf-8')}"
                    metadata = f"PNG: tEXt\nPositive prompt: {tEXt}"
                    return metadata
        return


def get_next_img(subctrl):
    if subctrl == 0:
        fimg_out = os.path.join(current_dir, txt2img_dir)
    elif subctrl == 1:
        fimg_out = os.path.join(current_dir, img2img_dir)
    files = os.listdir(fimg_out)
    png_files = [file for file in files if file.endswith('.png') and
                 file[:-4].isdigit()]
    if not png_files:
        return "1.png"
    highest_number = max([int(file[:-4]) for file in png_files])
    next_number = highest_number + 1
    fnext_img = f"{next_number}.png"
    return fnext_img


def txt2img(model, vae, taesd, cnnet, control_img, control_strength,
            ppromt, nprompt, sampling, steps, schedule, width, height,
            batch_count, cfg, seed, clip_skip, threads, vae_tiling,
            cont_net_cpu, rng, output, verbose):
    if model != "None":
        fmodel = os.path.join(current_dir, f'{model_dir}/{model}')
    if vae != "None":
        fvae = os.path.join(current_dir, f'{vae_dir}/{vae}')
    fembed = os.path.join(current_dir, f'{emb_dir}/')
    flora = os.path.join(current_dir, f'{lora_dir}/')
    if taesd:
        ftaesd = os.path.join(current_dir, f'{taesd_dir}/{taesd}')
    if cnnet:
        fcnnet = os.path.join(current_dir,
                              f'{cnnet_dir}/{cnnet}')
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
        foutput = os.path.join(current_dir, txt2img_dir + "/" +
                               get_next_img(subctrl=0))
    else:
        foutput = os.path.join(current_dir, f'"{txt2img_dir}/{output}.png"')
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
        command.extend(['--control-image', fcontrol_img])
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
    fmodel = os.path.join(current_dir, f'{model_dir}/{model}')
    if vae:
        fvae = os.path.join(current_dir, f'{vae_dir}/{vae}')
    fembed = os.path.join(current_dir, f'{emb_dir}/')
    flora = os.path.join(current_dir, f'{lora_dir}/')
    if taesd:
        ftaesd = os.path.join(current_dir, f'{taesd_dir}/{taesd}')
    fimg_inp = f'{img_inp}'
    if cnnet:
        fcnnet = os.path.join(current_dir,
                              f'{cnnet_dir}/{cnnet}')
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
        foutput = os.path.join(current_dir, img2img_dir + "/" +
                               get_next_img(subctrl=1))
    else:
        foutput = os.path.join(current_dir, f'"{img2img_dir}/{output}.png"')
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
        command.extend(['--control-image', fcontrol_img])
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


def convert(og_model, type, gguf_model, verbose):
    model_dir = os.path.join(current_dir, f"{model_dir}/")
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


def set_defaults(model, vae, sampling, steps, schedule, width, height):
    if model:
        data['def_model'] = model
    if vae:
        data['def_vae'] = vae
    data['def_sampling'] = sampling
    data['def_steps'] = steps
    data['def_schedule'] = schedule
    data['def_width'] = width
    data['def_height'] = height
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Set new defaults completed.")
    return


def rst_def():
    if 'def_model' in data:
        del data['def_model']
    if 'def_vae' in data:
        del data['def_vae']
    data['def_sampling'] = "euler_a"
    data['def_steps'] = 20
    data['def_schedule'] = "discrete"
    data['def_width'] = 512
    data['def_height'] = 512
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Reset defaults completed.")
    return


with gr.Blocks() as txt2img_block:
    txt2img_title = gr.Markdown("# Text to Image"),
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        rl_model = gr.Button(value=reload_symbol, scale=1)
        rl_model.click(reload_models, inputs=[model_dir_txt], outputs=[model])
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
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
                                                    "lcm"], value=def_sampling)
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule", choices=["discrete",
                                                                  "karras"],
                                       value=def_schedule)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=def_width, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=def_height, step=1)
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
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        rl_model = gr.Button(value=reload_symbol, scale=1)
        rl_model.click(reload_models, inputs=[model_dir_txt], outputs=[model])
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
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
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule",
                                       choices=["discrete", "karras"],
                                       value="discrete")
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=1, maximum=2048,
                                      value=def_width, step=1)
                    height = gr.Slider(label="Height", minimum=1, maximum=2048,
                                       value=def_height, step=1)
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


with gr.Blocks() as gallery_block:
    txt2img_ctrl = gr.Textbox(value=0, visible=False)
    img2img_ctrl = gr.Textbox(value=1, visible=False)
    with gr.Row():
        glr_txt2img = gr.Button(value="txt2img")
        glr_img2img = gr.Button(value="img2img")
    gallery_title = gr.Markdown('# Gallery')
    with gr.Row():
        glr_first = gr.Button(value="First page")
        glr_pvw = gr.Button(value="Previous")
        page_num_select = gr.Number(label="Page:", minimum=1, value=1,
                                    interactive=True)
        page_num_btn = gr.Button(value="Go")
        glr_nxt = gr.Button(value="Next")
        glr_last = gr.Button(value="End page")
    gallery = gr.Gallery(label="txt2img", columns=[4], rows=[4],
                         object_fit="contain", height="auto")
    img_info_txt = gr.Textbox(label="Metadata", value="", interactive=False)
    gallery.select(img_info, inputs=[], outputs=[img_info_txt])
    glr_txt2img.click(reload_gallery, inputs=[txt2img_ctrl],
                      outputs=[gallery, page_num_select])
    glr_img2img.click(reload_gallery, inputs=[img2img_ctrl],
                      outputs=[gallery, page_num_select])
    glr_pvw.click(prev_page, inputs=[], outputs=[gallery, page_num_select])
    glr_nxt.click(next_page, inputs=[], outputs=[gallery, page_num_select])
    glr_first.click(reload_gallery, inputs=[],
                    outputs=[gallery, page_num_select])
    glr_last.click(last_page, inputs=[],
                   outputs=[gallery, page_num_select])
    page_num_btn.click(goto_gallery, inputs=[page_num_select],
                       outputs=[gallery, page_num_select])


with gr.Blocks() as convert_block:
    convert_title = gr.Markdown("# Convert and Quantize")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                og_model = gr.Dropdown(label="Original Model",
                                       choices=get_hf_models(), scale=5)
                rl_model = gr.Button(reload_symbol, scale=1)
                rl_model.click(reload_hf_models, inputs=[], outputs=[og_model])
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


with gr.Blocks() as options_block:
    options_title = gr.Markdown("# Options")
    with gr.Column():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        with gr.Column(scale=1):
            rl_model = gr.Button(value=reload_symbol, scale=1)
            rl_model.click(reload_models, inputs=[model_dir_txt],
                           outputs=[model])
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            rl_vae = gr.Button(value=reload_symbol)
            rl_vae.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
            clear_vae = gr.ClearButton(vae)
        sampling = gr.Dropdown(label="Sampling method",
                               choices=["euler", "euler_a", "heun",
                                        "dpm2", "dpm2++2s_a",
                                        "dpm++2m", "dpm++2mv2",
                                        "lcm"], value=def_sampling)
        steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                          value=def_steps, step=1)
        schedule = gr.Dropdown(label="Schedule",
                               choices=["discrete", "karras"],
                               value="discrete")
        width = gr.Slider(label="Width", minimum=1, maximum=2048,
                          value=def_width, step=1)
        height = gr.Slider(label="Height", minimum=1, maximum=2048,
                           value=def_height, step=1)
        with gr.Row():
            set_btn = gr.Button(value="Set Defaults")
            set_btn.click(set_defaults, [model, vae, sampling, steps, schedule,
                                         width, height], [])
            rst_btn = gr.Button(value="Restore Defaults")
            rst_btn.click(rst_def, [], [])

sdcpp = gr.TabbedInterface(
    [txt2img_block, img2img_block, gallery_block, convert_block,
     options_block],
    ["txt2img", "img2img", "Gallery", "Checkpoint Converter", "Options"],
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

import os
import argparse
import subprocess
import gradio as gr
from PIL import Image
import json


json_path = 'config.json'
current_dir = os.getcwd()

samplers = ["euler", "euler_a", "heun", "dpm2", "dpm2++2s_a", "dpm++2m",
            "dpm++2mv2", "lcm"]
reload_symbol = '\U0001f504'
page_num = 0
ctrl = 0


with open(json_path, 'r') as json_file:
    data = json.load(json_file)


model_dir = eval(data['model_dir'])
vae_dir = eval(data['vae_dir'])
emb_dir = eval(data['emb_dir'])
lora_dir = eval(data['lora_dir'])
taesd_dir = eval(data['taesd_dir'])
cnnet_dir = eval(data['cnnet_dir'])
txt2img_dir = eval(data['txt2img_dir'])
img2img_dir = eval(data['img2img_dir'])


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
    fmodels_dir = model_dir
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(fmodels_dir + model) and
                (model.endswith((".gguf", ".safetensors", ".pth")))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def reload_models(model_dir):
    refreshed_models = gr.update(choices=get_models(model_dir))
    return refreshed_models


def get_hf_models():
    fmodels_dir = model_dir
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(fmodels_dir + model) and
                (model.endswith((".safetensors", ".ckpt", ".pth", ".gguf")))]
    else:
        print(f"The {fmodels_dir} folder does not exist.")
        return []


def reload_hf_models():
    refreshed_models = gr.update(choices=get_hf_models())
    return refreshed_models


def run_subprocess(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True) as process:

        # Read the output line by line in real-time
        for output_line in process.stdout:
            print(output_line.strip())

        # Wait for the process to finish and capture its errors and print them
        _, errors = process.communicate()
        if errors:
            print("Errors:", errors)


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
        image_path = img_dir + file_name
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
        image_path = img_dir + file_name
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


def extract_exif_from_jpg(img_path):
    if img_path.endswith(('.jpg', '.jpeg')):
        img = Image.open(img_path)
        exif_data = img._getexif()

        if exif_data is not None:
            user_comment = exif_data.get(37510)  # 37510 = UserComment tag
            if user_comment:
                return f"JPG: Exif\nPositive prompt: "\
                       f"{user_comment.decode('utf-8')[9::2]}"
            else:
                return "JPG: Exif\nPositive prompt: No User Comment found."
        else:
            return "JPG: Exif\nPositive prompt: No EXIF data found."
    else:
        return "Not a JPG image."


def img_info(sel_img: gr.SelectData):
    global ctrl
    global page_num
    img_index = (page_num * 16) - 16 + sel_img.index
    if ctrl == 0:
        img_dir = txt2img_dir
    elif ctrl == 1:
        img_dir = img2img_dir

    file_paths = [os.path.join(img_dir, file) for file in os.listdir(img_dir)
                  if os.path.isfile(os.path.join(img_dir, file)) and
                  file.lower().endswith(('.png', '.jpg'))]
    file_paths.sort()

    try:
        img_path = file_paths[img_index]
    except IndexError:
        print("Image index is out of range.")
        return

    if img_path.endswith(('.jpg', '.jpeg')):
        return extract_exif_from_jpg(img_path)

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
                    return f"PNG: tEXt\nPositive prompt: {tEXt}"
        return ""


def get_next_img(subctrl):
    if subctrl == 0:
        fimg_out = txt2img_dir
    elif subctrl == 1:
        fimg_out = img2img_dir
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
    fmodel = os.path.join(model_dir, model) if model else None
    fvae = os.path.join(vae_dir, vae) if vae else None
    ftaesd = os.path.join(taesd_dir, taesd) if taesd else None
    fcnnet = os.path.join(cnnet_dir, cnnet) if cnnet else None
    fcontrol_img = control_img if cnnet else None
    fcontrol_strength = str(control_strength) if cnnet else None
    fpprompt = f'"{ppromt}"'
    fnprompt = f'"{nprompt}"' if nprompt else None
    fsampling = str(sampling)
    fsteps = str(steps)
    fschedule = f'{schedule}'
    fwidth = str(width)
    fheight = str(height)
    fbatch_count = str(batch_count)
    fcfg = str(cfg)
    fseed = str(seed)
    fclip_skip = str(clip_skip + 1)
    fthreads = str(threads)
    fvae_tiling = vae_tiling if vae_tiling else None
    fcont_net_cpu = cont_net_cpu if cont_net_cpu else None
    frng = str(rng)
    foutput = (os.path.join(txt2img_dir, f"{output}.png")
               if output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    command = [sd, '-M', 'txt2img', '-m', fmodel, '-p', fpprompt,
               '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--cfg-scale', fcfg, '-s', fseed, '--clip-skip',
               fclip_skip, '--embd-dir', emb_dir, '--lora-model-dir', lora_dir,
               '-t', fthreads, '--rng', frng, '-o', foutput]

    if fvae:
        command.extend(['--vae', fvae])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
    if fcnnet:
        command.extend(['--control-net', fcnnet, '--control-image',
                        fcontrol_img, '--control-strength', fcontrol_strength])
    if fnprompt:
        command.extend(['-n', fnprompt])
    if fvae_tiling:
        command.extend(['--vae-tiling'])
    if fcont_net_cpu:
        command.extend(['--cont_net_cpu'])
    if verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def img2img(model, vae, taesd, img_inp, cnnet, control_img,
            control_strength, ppromt, nprompt, sampling, steps, schedule,
            width, height, batch_count, strenght, cfg, seed, clip_skip,
            threads, vae_tiling, cont_net_cpu, rng, output, verbose):
    fmodel = os.path.join(model_dir, model) if model else None
    fvae = os.path.join(vae_dir, vae) if vae else None
    ftaesd = os.path.join(taesd_dir, taesd) if taesd else None
    fcnnet = os.path.join(cnnet_dir, cnnet) if cnnet else None
    fcontrol_img = control_img if cnnet else None
    fcontrol_strength = str(control_strength) if cnnet else None
    fpprompt = f'"{ppromt}"'
    fnprompt = f'"{nprompt}"' if nprompt else None
    fsampling = str(sampling)
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
    fvae_tiling = vae_tiling if vae_tiling else None
    fcont_net_cpu = cont_net_cpu if cont_net_cpu else None
    frng = str(rng)
    foutput = (os.path.join(img2img_dir, f"{output}.png")
               if output
               else os.path.join(img2img_dir, get_next_img(subctrl=1)))

    command = [sd, '-M', 'img2img', '-m', fmodel, '-i', img_inp, '-p',
               fpprompt, '--sampling-method', fsampling, '--steps', fsteps,
               '--schedule', fschedule, '-W', fwidth, '-H', fheight, '-b',
               fbatch_count, '--strength', fstrenght, '--cfg-scale', fcfg,
               '-s', fseed, '--clip-skip', fclip_skip, '--embd-dir', emb_dir,
               '--lora-model-dir', lora_dir, '-t', fthreads, '--rng', frng,
               '-o', foutput]

    if fvae:
        command.extend(['--vae', fvae])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
    if fcnnet:
        command.extend(['--control-net', fcnnet, '--control-image',
                        fcontrol_img, '--control-strength', fcontrol_strength])
    if fnprompt:
        command.extend(['-n', fnprompt])
    if fvae_tiling:
        command.extend(['--vae-tiling'])
    if fcont_net_cpu:
        command.extend(['--cont_net_cpu'])
    if verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def convert(orig_model, quant_type, gguf_name, verbose):
    forig_model = os.path.join(model_dir, orig_model)
    if not gguf_name:
        model_name, ext = os.path.splitext(orig_model)
        fgguf_name = f"{os.path.join(model_dir, model_name)}"\
                     f".{quant_type}.gguf"
    else:
        fgguf_name = os.path.join(model_dir, gguf_name)

    command = [sd, '-M', 'convert', '-m', forig_model,
               '-o', fgguf_name, '--type', quant_type]

    if verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return "Process completed."


def set_defaults(model, vae, sampling, steps, schedule, width, height,
                 model_dir_txt, vae_dir_txt, emb_dir_txt, lora_dir_txt,
                 taesd_dir_txt, cnnet_dir_txt, txt2img_dir_txt,
                 img2img_dir_txt):
    data.update({
        'model_dir': model_dir_txt,
        'vae_dir': vae_dir_txt,
        'emb_dir': emb_dir_txt,
        'lora_dir': lora_dir_txt,
        'taesd_dir': taesd_dir_txt,
        'cnnet_dir': cnnet_dir_txt,
        'txt2img_dir': txt2img_dir_txt,
        'img2img_dir': img2img_dir_txt,
        'def_sampling': sampling,
        'def_steps': steps,
        'def_schedule': schedule,
        'def_width': width,
        'def_height': height
    })

    if model:
        data['def_model'] = model
    if vae:
        data['def_vae'] = vae

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print("Set new defaults completed.")
    return


def rst_def():
    data.update({
        'model_dir': 'os.path.join(current_dir, "models/Stable-Diffusion/")',
        'vae_dir': 'os.path.join(current_dir, "models/VAE/")',
        'emb_dir': 'os.path.join(current_dir, "models/Embeddings/")',
        'lora_dir': 'os.path.join(current_dir, "models/Lora/")',
        'taesd_dir': 'os.path.join(current_dir, "models/TAESD/")',
        'cnnet_dir': 'os.path.join(current_dir, "models/ControlNet/")',
        'txt2img_dir': 'os.path.join(current_dir, "outputs/txt2img/")',
        'img2img_dir': 'os.path.join(current_dir, "outputs/img2img/")',
        'def_sampling': "euler_a",
        'def_steps': 20,
        'def_schedule': "discrete",
        'def_width': 512,
        'def_height': 512
    })

    data.pop('def_model', None)
    data.pop('def_vae', None)

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print("Reset defaults completed.")
    return


with gr.Blocks() as txt2img_block:
    # Directory Textboxes
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    txt2img_title = gr.Markdown("# Text to Image"),

    # Model & VAE Selection
    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        reload_model_btn = gr.Button(value=reload_symbol, scale=1)
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=reload_symbol)
            clear_vae = gr.ClearButton(vae)

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=reload_symbol, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
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

    # Settings
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=samplers,
                                           value=def_sampling)
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule", choices=["discrete",
                                                                  "karras"],
                                       value=def_schedule)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=2048,
                                      value=def_width, step=8)
                    height = gr.Slider(label="Height", minimum=64,
                                       maximum=2048, value=def_height, step=8)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, value=1, step=1)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            value=7.0, step=0.1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=reload_symbol)
                clear_cnnet = gr.ClearButton(cnnet)
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
                cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")

            # Extra Settings
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                vae_tiling = gr.Checkbox(label="Vae Tiling")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                output = gr.Textbox(label="Output Name",
                                    placeholder="Optional")
                verbose = gr.Checkbox(label="Verbose")

        # Output
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")

    # Generate
    gen_btn.click(txt2img, inputs=[model, vae, taesd, cnnet,
                                   control_img, control_strength,
                                   pprompt, nprompt, sampling, steps,
                                   schedule, width, height,
                                   batch_count, cfg, seed, clip_skip,
                                   threads, vae_tiling, cnnet_cpu,
                                   rng, output, verbose], outputs=[img_final])

    # Interactive Bindings
    reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                           outputs=[model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_cnnet_btn.click(reload_models, inputs=[cnnet_dir_txt],
                           outputs=[cnnet])


with gr.Blocks()as img2img_block:
    # Directory Textboxes
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    img2img_title = gr.Markdown("# Image to Image")

    # Model & VAE Selection
    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        reload_model_btn = gr.Button(value=reload_symbol, scale=1)
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=reload_symbol)
            clear_vae = gr.ClearButton(vae)

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=reload_symbol, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
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

    # Settings
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=1):
                    sampling = gr.Dropdown(label="Sampling method",
                                           choices=samplers, value="euler_a")
                with gr.Column(scale=1):
                    steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                                      value=def_steps, step=1)
            with gr.Row():
                schedule = gr.Dropdown(label="Schedule",
                                       choices=["discrete", "karras"],
                                       value="discrete")
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=2048,
                                      value=def_width, step=8)
                    height = gr.Slider(label="Height", minimum=64,
                                       maximum=2048, value=def_height, step=8)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, step=1, value=1)
            strenght = gr.Slider(label="Noise strenght", minimum=0, maximum=1,
                                 step=0.01, value=0.75)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            step=0.1, value=7.0)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=reload_symbol)
                clear_connet = gr.ClearButton(cnnet)
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
                cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")

            # Extra Settings
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                vae_tiling = gr.Checkbox(label="Vae Tiling")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                output = gr.Textbox(label="Output Name (optional)", value="")
                verbose = gr.Checkbox(label="Verbose")
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")

    # Generate
    gen_btn.click(img2img, inputs=[model, vae, taesd, img_inp,
                                   cnnet, control_img,
                                   control_strength, pprompt,
                                   nprompt, sampling, steps, schedule,
                                   width, height, batch_count,
                                   strenght, cfg, seed, clip_skip,
                                   threads, vae_tiling, cnnet_cpu,
                                   rng, output, verbose], outputs=[img_final])

    # Interactive Bindings
    reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                           outputs=[model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_cnnet_btn.click(reload_models, inputs=[cnnet_dir_txt],
                           outputs=[cnnet])


with gr.Blocks() as gallery_block:
    # Controls
    txt2img_ctrl = gr.Textbox(value=0, visible=False)
    img2img_ctrl = gr.Textbox(value=1, visible=False)

    # Title
    gallery_title = gr.Markdown('# Gallery')

    # Gallery Navigation Buttons
    with gr.Row():
        txt2img_btn = gr.Button(value="txt2img")
        img2img_btn = gr.Button(value="img2img")

    with gr.Row():
        pvw_btn = gr.Button(value="Previous")
        page_num_select = gr.Number(label="Page:", minimum=1, value=1,
                                    interactive=True)
        go_btn = gr.Button(value="Go")
        nxt_btn = gr.Button(value="Next")

    with gr.Row():
        first_btn = gr.Button(value="First page")
        last_btn = gr.Button(value="Last page")

    # Gallery Display
    gallery = gr.Gallery(label="txt2img", columns=[4], rows=[4],
                         object_fit="contain", height="auto")

    # Image Information Display
    img_info_txt = gr.Textbox(label="Metadata", value="", interactive=False)

    # Interactive bindings
    gallery.select(img_info, inputs=[], outputs=[img_info_txt])
    txt2img_btn.click(reload_gallery, inputs=[txt2img_ctrl],
                      outputs=[gallery, page_num_select])
    img2img_btn.click(reload_gallery, inputs=[img2img_ctrl],
                      outputs=[gallery, page_num_select])
    pvw_btn.click(prev_page, inputs=[], outputs=[gallery, page_num_select])
    nxt_btn.click(next_page, inputs=[], outputs=[gallery, page_num_select])
    first_btn.click(reload_gallery, inputs=[],
                    outputs=[gallery, page_num_select])
    last_btn.click(last_page, inputs=[],
                   outputs=[gallery, page_num_select])
    go_btn.click(goto_gallery, inputs=[page_num_select],
                 outputs=[gallery, page_num_select])


with gr.Blocks() as convert_block:
    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        # Input
        with gr.Column(scale=1):
            with gr.Row():
                orig_model = gr.Dropdown(label="Original Model",
                                         choices=get_hf_models(), scale=5)
                reload_btn = gr.Button(reload_symbol, scale=1)
                reload_btn.click(reload_hf_models, inputs=[],
                                 outputs=[orig_model])

            quant_type = gr.Dropdown(label="Type",
                                     choices=["f32", "f16", "q8_0", "q5_1",
                                              "q5_0", "q4_1", "q4_0"],
                                     value="f32")

            verbose = gr.Checkbox(label="Verbose")

            gguf_name = gr.Textbox(label="Output Name (optional, must end "
                                   "with .gguf)", value="")

            convert_btn = gr.Button(value="Convert")

        # Output
        with gr.Column(scale=1):
            result = gr.Textbox(interactive=False, value="")

    # Interactive Bindings
    convert_btn.click(convert, inputs=[orig_model, quant_type, gguf_name,
                                       verbose], outputs=[result])


with gr.Blocks() as options_block:
    # Title
    options_title = gr.Markdown("# Options")

    with gr.Column():
        # Model Dropdown
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        with gr.Column(scale=1):
            reload_model_btn = gr.Button(value=reload_symbol, scale=1)
            reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                                   outputs=[model])

        # VAE Dropdown
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=reload_symbol)
            reload_vae_btn.click(reload_models, inputs=[vae_dir_txt],
                                 outputs=[vae])
            clear_vae = gr.ClearButton(vae)

        # Sampling Method Dropdown
        sampling = gr.Dropdown(label="Sampling method",
                               choices=samplers, value=def_sampling)

        # Steps Slider
        steps = gr.Slider(label="Steps", minimum=1, maximum=99,
                          value=def_steps, step=1)

        # Schedule Dropdown
        schedule = gr.Dropdown(label="Schedule",
                               choices=["discrete", "karras"],
                               value="discrete")

        # Size Sliders
        width = gr.Slider(label="Width", minimum=64, maximum=2048,
                          value=def_width, step=8)
        height = gr.Slider(label="Height", minimum=64, maximum=2048,
                           value=def_height, step=8)

        # Folders Accordion
        with gr.Accordion(label="Folders", open=False):
            model_dir_txt = gr.Textbox(label="Models folder", value=model_dir,
                                       interactive=True)
            vae_dir_txt = gr.Textbox(label="VAE folder", value=vae_dir,
                                     interactive=True)
            emb_dir_txt = gr.Textbox(label="Embeddings folder", value=emb_dir,
                                     interactive=True)
            lora_dir_txt = gr.Textbox(label="Lora folder", value=lora_dir,
                                      interactive=True)
            taesd_dir_txt = gr.Textbox(label="TAESD folder", value=taesd_dir,
                                       interactive=True)
            cnnet_dir_txt = gr.Textbox(label="ControlNet folder",
                                       value=cnnet_dir, interactive=True)
            txt2img_dir_txt = gr.Textbox(label="txt2img outputs folder",
                                         value=txt2img_dir, interactive=True)
            img2img_dir_txt = gr.Textbox(label="img2img outputs folder",
                                         value=img2img_dir, interactive=True)

        # Set Defaults and Restore Defaults Buttons
        with gr.Row():
            set_btn = gr.Button(value="Set Defaults")
            set_btn.click(set_defaults, [model, vae, sampling, steps, schedule,
                                         width, height, model_dir_txt,
                                         vae_dir_txt, emb_dir_txt,
                                         lora_dir_txt, taesd_dir_txt,
                                         cnnet_dir_txt, txt2img_dir_txt,
                                         img2img_dir_txt], [])
            restore_btn = gr.Button(value="Restore Defaults")
            restore_btn.click(rst_def, [], [])


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

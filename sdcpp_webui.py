"""sd.cpp-webui - Main module"""

import os
import argparse
import subprocess
import json
from PIL import Image
import gradio as gr


def main():
    """Main"""
    parser = argparse.ArgumentParser(description='Process optional arguments')
    parser.add_argument('--listen', action='store_true',
                        help='Listen on 0.0.0.0')
    parser.add_argument('--autostart', action='store_true',
                        help='Automatically launch in a new browser tab')
    args = parser.parse_args()
    sdcpp_launch(args.listen, args.autostart)


def sdcpp_launch(listen=False, autostart=False):
    """Logic for launching sdcpp based on arguments"""
    if listen and autostart:
        print("Launching sdcpp with --listen --autostart")
        sdcpp.launch(server_name="0.0.0.0", inbrowser=True)
    elif listen:
        print("Launching sdcpp with --listen")
        sdcpp.launch(server_name="0.0.0.0")
    elif autostart:
        print("Launching sdcpp with --autostart")
        sdcpp.launch(inbrowser=True)
    else:
        print("Launching sdcpp without any specific options")
        sdcpp.launch()


def get_models(models_folder):
    """Lists models in a folder"""
    if os.path.isdir(models_folder):
        models = [model for model in os.listdir(models_folder)
                 if os.path.isfile(models_folder + model) and
                 (model.endswith((".gguf", ".safetensors", ".pth")))]
        return models

    print(f"The {models_folder} folder does not exist.")
    return []


def reload_models(models_folder):
    """Reloads models list"""
    refreshed_models = gr.update(choices=get_models(models_folder))
    return refreshed_models


def get_prompts():
    """Lists saved prompts"""
    with open('prompts.json', 'r') as file:
        prompts_data = json.load(file)

    prompts_keys = list(prompts_data.keys())

    return prompts_keys


def save_prompts(prompts, pprompt, nprompt):
    """Saves a prompt"""
    if prompts is not None and prompts.strip():
        with open('prompts.json', 'r') as file:
            prompts_data = json.load(file)

        prompts_data[prompts] = {
            'positive': pprompt,
            'negative': nprompt
        }

        with open('prompts.json', 'w') as file:
            json.dump(prompts_data, file, indent=4)


def delete_prompts(prompt):
    """Deletes a saved prompt"""
    with open('prompts.json', 'r') as file:
        prompts_data = json.load(file)

    if prompt in prompts_data:
        del prompts_data[prompt]
        print(f"Key '{prompt}' deleted.")
    else:
        print(f"Key '{prompt}' not found.")
    
    with open('prompts.json', 'w') as file:
        json.dump(prompts_data, file, indent=4)

    return


def reload_prompts():
    """Reloads prompts list"""
    refreshed_prompts = gr.update(choices=get_prompts())
    return refreshed_prompts


def load_prompts(prompt):
    """Loads a saved prompt"""
    with open('prompts.json', 'r') as file:
        prompts_data = json.load(file)
    positive_prompts = []
    negative_prompts = []
    key_data = prompts_data.get(prompt, {})
    positive_prompts = key_data.get('positive', '')
    negative_prompts = key_data.get('negative', '')

    pprompt_load = gr.update(value=positive_prompts)
    nprompt_load = gr.update(value=negative_prompts)
    return pprompt_load, nprompt_load


def get_hf_models():
    """Lists convertible models in a folder"""
    fmodels_dir = model_dir
    if os.path.isdir(fmodels_dir):
        return [model for model in os.listdir(fmodels_dir)
                if os.path.isfile(fmodels_dir + model) and
                (model.endswith((".safetensors", ".ckpt", ".pth", ".gguf")))]

    print(f"The {fmodels_dir} folder does not exist.")
    return []


def reload_hf_models():
    """Reloads convertible models list"""
    refreshed_models = gr.update(choices=get_hf_models())
    return refreshed_models


def run_subprocess(command):
    """Runs subprocess"""
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
    """Reloads gallery_block"""
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
    start_index = (fpage_num * 16) - 16
    end_index = min(start_index + 16, len(image_files))
    for file_name in image_files[start_index:end_index]:
        image_path = img_dir + file_name
        image = Image.open(image_path)
        imgs.append(image)
    page_num = fpage_num
    if subctrl == 0:
        return imgs, page_num, gr.Gallery(selected_index=None)

    return imgs


def goto_gallery(fpage_num=1):
    """Loads a specific gallery page"""
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
    page_num = fpage_num if fpage_num < total_pages else total_pages
    files = os.listdir(img_dir)
    image_files = [file for file in files if file.endswith(('.jpg', '.png'))]
    start_index = (page_num * 16) - 16
    end_index = min(start_index + 16, len(image_files))
    for file_name in image_files[start_index:end_index]:
        image_path = img_dir + file_name
        image = Image.open(image_path)
        imgs.append(image)
    return imgs, page_num, gr.Gallery(selected_index=None)


def next_page():
    """Moves to the next gallery page"""
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
    return imgs, page_num, gr.Gallery(selected_index=None)


def prev_page():
    """Moves to the previous gallery page"""
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
    return imgs, page_num, gr.Gallery(selected_index=None)


def last_page():
    """Moves to the last gallery page"""
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
    return imgs, page_num, gr.Gallery(selected_index=None)


def extract_exif_from_jpg(img_path):
    """Extracts exif data from jpg"""
    img = Image.open(img_path)
    exif_data = img._getexif()

    if exif_data is not None:
        user_comment = exif_data.get(37510)  # 37510 = UserComment tag
        if user_comment:
            return f"JPG: Exif\nPositive prompt: "\
                   f"{user_comment.decode('utf-8')[9::2]}"

        return "JPG: No User Comment found."

    return "JPG: Exif\nNo EXIF data found."


def img_info(sel_img: gr.SelectData):
    """Reads generation data from an image"""
    img_index = (page_num * 16) - 16 + sel_img.index
    if ctrl == 0:
        img_dir = txt2img_dir
    elif ctrl == 1:
        img_dir = img2img_dir

    file_paths = [os.path.join(img_dir, file) for file in os.listdir(img_dir)
                  if os.path.isfile(os.path.join(img_dir, file)) and
                  file.lower().endswith(('.png', '.jpg'))]
    file_paths.sort(key=os.path.getctime)

    try:
        img_path = file_paths[img_index]
    except IndexError:
        return print("Image index is out of range.")

    if img_path.endswith(('.jpg', '.jpeg')):
        return extract_exif_from_jpg(img_path)

    if img_path.endswith('.png'):
        with open(img_path, 'rb') as file:
            if file.read(8) != b'\x89PNG\r\n\x1a\n':
                return None
            while True:
                length_chunk = file.read(4)
                if not length_chunk:
                    return None
                length = int.from_bytes(length_chunk, byteorder='big')
                chunk_type = file.read(4).decode('utf-8')
                png_block = file.read(length)
                _ = file.read(4)
                if chunk_type == 'tEXt':
                    _, value = png_block.split(b'\x00', 1)
                    png_exif = f"{value.decode('utf-8')}"
                    return f"PNG: tEXt\nPositive prompt: {png_exif}"
    return None


def get_next_img(subctrl):
    """Creates a new image name"""
    if subctrl == 0:
        fimg_out = txt2img_dir
    elif subctrl == 1:
        fimg_out = img2img_dir
    files = os.listdir(fimg_out)
    png_files = [file for file in files if file.endswith('.png') and
                 file[:-4].isdigit()]
    if not png_files:
        return "1.png"
    highest_number = max(int(file[:-4]) for file in png_files)
    next_number = highest_number + 1
    fnext_img = f"{next_number}.png"
    return fnext_img


def txt2img(in_model, in_vae, in_taesd, in_upscl, in_cnnet, in_control_img,
            in_control_strength, in_ppromt, in_nprompt, in_sampling,
            in_steps, in_schedule, in_width, in_height, in_batch_count,
            in_cfg, in_seed, in_clip_skip, in_threads, in_vae_tiling,
            in_cnnet_cpu, in_rng, in_output, in_color, in_verbose):
    """Text to image command creator"""
    fmodel = os.path.join(model_dir, in_model) if in_model else None
    fvae = os.path.join(vae_dir, in_vae) if in_vae else None
    ftaesd = os.path.join(taesd_dir, in_taesd) if in_taesd else None
    fupscl = os.path.join(upscl_dir, in_upscl) if in_upscl else None
    fcnnet = os.path.join(cnnet_dir, in_cnnet) if in_cnnet else None
    foutput = (os.path.join(txt2img_dir, f"{in_output}.png")
               if in_output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    command = [sd, '-M', 'txt2img', '-m', fmodel, '-p', f'"{in_ppromt}"',
               '--sampling-method', str(in_sampling), '--steps', str(in_steps),
               '--schedule', f'{in_schedule}', '-W', str(in_width), '-H',
               str(in_height), '-b', str(in_batch_count), '--cfg-scale',
               str(in_cfg), '-s', str(in_seed), '--clip-skip',
               str(in_clip_skip + 1), '--embd-dir', emb_dir,
               '--lora-model-dir', lora_dir, '-t', str(in_threads), '--rng',
               str(in_rng), '-o', foutput]

    if fvae:
        command.extend(['--vae', fvae])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
    if fupscl:
        command.extend(['--upscale-model', fupscl])
    if fcnnet:
        command.extend(['--control-net', fcnnet, '--control-image',
                        in_control_img, '--control-strength',
                        str(in_control_strength)])
    if in_nprompt:
        command.extend(['-n', f'"{in_nprompt}"'])
    if in_vae_tiling:
        command.extend(['--vae-tiling'])
    if in_cnnet_cpu:
        command.extend(['--control-net-cpu'])
    if in_color:
        command.extend(['--color'])
    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def img2img(in_model, in_vae, in_taesd, in_img_inp, in_upscl, in_cnnet,
            in_control_img, in_control_strength, in_ppromt, in_nprompt,
            in_sampling, in_steps, in_schedule, in_width, in_height,
            in_batch_count, in_strenght, in_cfg, in_seed, in_clip_skip,
            in_threads, in_vae_tiling, in_cnnet_cpu, in_canny, in_rng,
            in_output, in_color, in_verbose):
    """Image to image command creator"""
    fmodel = os.path.join(model_dir, in_model) if in_model else None
    fvae = os.path.join(vae_dir, in_vae) if in_vae else None
    ftaesd = os.path.join(taesd_dir, in_taesd) if in_taesd else None
    fupscl = os.path.join(upscl_dir, in_upscl) if in_upscl else None
    fcnnet = os.path.join(cnnet_dir, in_cnnet) if in_cnnet else None
    foutput = (os.path.join(img2img_dir, f"{in_output}.png")
               if in_output
               else os.path.join(img2img_dir, get_next_img(subctrl=1)))

    command = [sd, '-M', 'img2img', '-m', fmodel, '-i', in_img_inp, '-p',
               f'"{in_ppromt}"', '--sampling-method', str(in_sampling),
               '--steps', str(in_steps), '--schedule', f'{in_schedule}', '-W',
               str(in_width), '-H', str(in_height), '-b', str(in_batch_count),
               '--strength', str(in_strenght), '--cfg-scale', str(in_cfg),
               '-s', str(in_seed), '--clip-skip', str(in_clip_skip + 1),
               '--embd-dir', emb_dir, '--lora-model-dir', lora_dir, '-t',
               str(in_threads), '--rng', str(in_rng), '-o', foutput]

    if fvae:
        command.extend(['--vae', fvae])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
    if fupscl:
        command.extend(['--upscale-model', fupscl])
    if fcnnet:
        command.extend(['--control-net', fcnnet, '--control-image',
                        in_control_img, '--control-strength',
                        str(in_control_strength)])
    if in_nprompt:
        command.extend(['-n', f'"{in_nprompt}"'])
    if in_vae_tiling:
        command.extend(['--vae-tiling'])
    if in_cnnet_cpu:
        command.extend(['--control-net-cpu'])
    if in_canny:
        command.extend(['--canny'])
    if in_color:
        command.extend(['--color'])
    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def convert(in_orig_model, in_quant_type, in_gguf_name, in_verbose):
    """Convert model command creator"""
    forig_model = os.path.join(model_dir, in_orig_model)
    if not in_gguf_name:
        model_name, _ = os.path.splitext(in_orig_model)
        fgguf_name = f"{os.path.join(model_dir, model_name)}"\
                     f".{in_quant_type}.gguf"
    else:
        fgguf_name = os.path.join(model_dir, in_gguf_name)

    command = [sd, '-M', 'convert', '-m', forig_model,
               '-o', fgguf_name, '--type', in_quant_type]

    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return "Process completed."


def set_defaults(in_model, in_vae, in_sampling, in_steps, in_schedule,
                 in_width, in_height, in_model_dir_txt, in_vae_dir_txt,
                 in_emb_dir_txt, in_lora_dir_txt, in_taesd_dir_txt,
                 in_upscl_dir_txt, in_cnnet_dir_txt, in_txt2img_dir_txt,
                 in_img2img_dir_txt):
    """Sets new defaults"""
    data.update({
        'model_dir': in_model_dir_txt,
        'vae_dir': in_vae_dir_txt,
        'emb_dir': in_emb_dir_txt,
        'lora_dir': in_lora_dir_txt,
        'taesd_dir': in_taesd_dir_txt,
        'upscl_dir' : in_upscl_dir_txt,
        'cnnet_dir': in_cnnet_dir_txt,
        'txt2img_dir': in_txt2img_dir_txt,
        'img2img_dir': in_img2img_dir_txt,
        'def_sampling': in_sampling,
        'def_steps': in_steps,
        'def_schedule': in_schedule,
        'def_width': in_width,
        'def_height': in_height
    })

    if in_model:
        data['def_model'] = in_model
    if in_vae:
        data['def_vae'] = in_vae

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(data, json_file_w, indent=4)

    print("Set new defaults completed.")


def rst_def():
    """Restores factory defaults"""
    data.update({
        'model_dir': 'os.path.join(current_dir, "models/Stable-Diffusion/")',
        'vae_dir': 'os.path.join(current_dir, "models/VAE/")',
        'emb_dir': 'os.path.join(current_dir, "models/Embeddings/")',
        'lora_dir': 'os.path.join(current_dir, "models/Lora/")',
        'taesd_dir': 'os.path.join(current_dir, "models/TAESD/")',
        'upscl_dir' : 'os.path.join(current_dir, "models/Upscalers/")',
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

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(data, json_file_w, indent=4)

    print("Reset defaults completed.")


CONFIG_PATH = 'config.json'
current_dir = os.getcwd()

samplers = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m",
            "dpm++2mv2", "lcm"]
schedulers = ["discrete", "karras", "ays"]
RELOAD_SYMBOL = '\U0001f504'
page_num = 0
ctrl = 0

if not os.path.isfile('config.json'):
        # Create an empty JSON file
    with open('config.json', 'w') as file:
        # Write an empty JSON object
        json.dump({}, file, indent=4)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as json_file_r:
        data = json.load(json_file_r)
    rst_def()
    print(f"File 'config.json' created and initialized.")

with open(CONFIG_PATH, 'r', encoding='utf-8') as json_file_r:
    data = json.load(json_file_r)


model_dir = eval(data['model_dir'])
vae_dir = eval(data['vae_dir'])
emb_dir = eval(data['emb_dir'])
lora_dir = eval(data['lora_dir'])
taesd_dir = eval(data['taesd_dir'])
upscl_dir = eval(data['upscl_dir'])
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

if not os.path.isfile('prompts.json'):
        # Create an empty JSON file
    with open('prompts.json', 'w') as file:
        # Write an empty JSON object
        json.dump({}, file, indent=4)
    print(f"File 'prompts.json' created and initialized as an empty JSON file.")


if not os.system("which lspci > /dev/null") == 0:
    if os.name == "nt":
        sd = "sd.exe"
    elif os.name == "posix":
        sd = "./sd"
else:
    sd = "./sd"


with gr.Blocks() as txt2img_block:
    # Directory Textboxes
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    txt2img_title = gr.Markdown("# Text to Image")

    # Model & VAE Selection
    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        reload_model_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=RELOAD_SYMBOL)
            clear_vae = gr.ClearButton(vae)

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
    with gr.Row():
        with gr.Accordion(label="Saved prompts", open=False):
            saved_prompts = gr.Dropdown(label="Prompts",
                                        choices=get_prompts(),
                                        interactive=True,
                                        allow_custom_value=True)
            with gr.Row():
                save_prompt_btn = gr.Button(value="Save prompt", size="lg")
                del_prompt_btn = gr.Button(value="Delete prompt", size="lg")
                reload_prompts_btn = gr.Button(value=RELOAD_SYMBOL)
            with gr.Row():
                load_prompt_btn = gr.Button(value="Load prompt", size="lg")
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
                schedule = gr.Dropdown(label="Schedule", choices=schedulers,
                                       value=def_schedule)
            with gr.Row():
                with gr.Column():
                    width = gr.Slider(label="Width", minimum=64, maximum=2048,
                                      value=def_width, step=64)
                    height = gr.Slider(label="Height", minimum=64,
                                       maximum=2048, value=def_height, step=64)
                batch_count = gr.Slider(label="Batch count", minimum=1,
                                        maximum=99, value=1, step=1)
            cfg = gr.Slider(label="CFG Scale", minimum=1, maximum=30,
                            value=7.0, step=0.1)
            seed = gr.Number(label="Seed", minimum=-1, maximum=2**32, value=-1)
            clip_skip = gr.Slider(label="CLIP skip", minimum=0, maximum=12,
                                  value=0, step=0.1)

            # Upscale
            with gr.Accordion(label="Upscale", open=False):
                upscl = gr.Dropdown(label="Upscaler",
                                    choices=get_models(upscl_dir))
                reload_upscl_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_upscl = gr.ClearButton(upscl)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=RELOAD_SYMBOL)
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
                color = gr.Checkbox(label="Color", value=True)
                verbose = gr.Checkbox(label="Verbose")

        # Output
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")

    # Generate
    gen_btn.click(txt2img, inputs=[model, vae, taesd, upscl, cnnet,
                                   control_img, control_strength,
                                   pprompt, nprompt, sampling, steps,
                                   schedule, width, height, batch_count,
                                   cfg, seed, clip_skip, threads,
                                   vae_tiling, cnnet_cpu, rng, output,
                                   color, verbose],
                  outputs=[img_final])

    # Interactive Bindings
    reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                           outputs=[model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_upscl_btn.click(reload_models, inputs=[upscl_dir_txt],
                           outputs=[upscl])
    reload_cnnet_btn.click(reload_models, inputs=[cnnet_dir_txt],
                           outputs=[cnnet])
    save_prompt_btn.click(save_prompts, inputs=[saved_prompts, pprompt,
                                                nprompt], outputs=[])
    del_prompt_btn.click(delete_prompts, inputs=[saved_prompts],
                         outputs=[])
    reload_prompts_btn.click(reload_prompts, inputs=[],
                             outputs=[saved_prompts])
    load_prompt_btn.click(load_prompts, inputs=[saved_prompts],
                          outputs=[pprompt, nprompt])


with gr.Blocks()as img2img_block:
    # Directory Textboxes
    model_dir_txt = gr.Textbox(value=model_dir, visible=False)
    vae_dir_txt = gr.Textbox(value=vae_dir, visible=False)
    emb_dir_txt = gr.Textbox(value=emb_dir, visible=False)
    lora_dir_txt = gr.Textbox(value=lora_dir, visible=False)
    taesd_dir_txt = gr.Textbox(value=taesd_dir, visible=False)
    upscl_dir_txt = gr.Textbox(value=upscl_dir, visible=False)
    cnnet_dir_txt = gr.Textbox(value=cnnet_dir, visible=False)

    # Title
    img2img_title = gr.Markdown("# Image to Image")

    # Model & VAE Selection
    with gr.Row():
        model = gr.Dropdown(label="Model",
                            choices=get_models(model_dir), scale=7,
                            value=def_model)
        reload_model_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=RELOAD_SYMBOL)
            clear_vae = gr.ClearButton(vae)

    # Extra Networks Selection
    with gr.Row():
        with gr.Accordion(label="Extra Networks", open=False):
            with gr.Row():
                taesd = gr.Dropdown(label="TAESD",
                                    choices=get_models(taesd_dir), scale=7)
                with gr.Column():
                    reload_taesd_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
                    clear_taesd = gr.ClearButton(taesd, scale=1)

    # Prompts
    with gr.Row():
        with gr.Accordion(label="Saved prompts", open=False):
            saved_prompts = gr.Dropdown(label="Prompts",
                                        choices=get_prompts(),
                                        interactive=True,
                                        allow_custom_value=True)
            with gr.Row():
                save_prompt_btn = gr.Button(value="Save prompt", size="lg")
                del_prompt_btn = gr.Button(value="Delete prompt", size="lg")
                reload_prompts_btn = gr.Button(value=RELOAD_SYMBOL)
            with gr.Row():
                load_prompt_btn = gr.Button(value="Load prompt", size="lg")
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
                                       choices=schedulers,
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

            # Upscale
            with gr.Accordion(label="Upscale", open=False):
                upscl = gr.Dropdown(label="Upscaler",
                                    choices=get_models(upscl_dir))
                reload_upscl_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_upscl = gr.ClearButton(upscl)

            # ControlNet
            with gr.Accordion(label="ControlNet", open=False):
                cnnet = gr.Dropdown(label="ControlNet",
                                    choices=get_models(cnnet_dir))
                reload_cnnet_btn = gr.Button(value=RELOAD_SYMBOL)
                clear_connet = gr.ClearButton(cnnet)
                control_img = gr.Image(sources="upload", type="filepath")
                control_strength = gr.Slider(label="ControlNet strength",
                                             minimum=0, maximum=1, step=0.01,
                                             value=0.9)
                cnnet_cpu = gr.Checkbox(label="ControlNet on CPU")
                canny = gr.Checkbox(label="Canny (edge detection)")

            # Extra Settings
            with gr.Accordion(label="Extra", open=False):
                threads = gr.Number(label="Threads", minimum=0,
                                    maximum=os.cpu_count(), value=0)
                vae_tiling = gr.Checkbox(label="Vae Tiling")
                rng = gr.Dropdown(label="RNG", choices=["std_default", "cuda"],
                                  value="cuda")
                output = gr.Textbox(label="Output Name (optional)", value="")
                color = gr.Checkbox(label="Color", value=True)
                verbose = gr.Checkbox(label="Verbose")
        with gr.Column(scale=1):
            img_final = gr.Gallery(label="Generated images", show_label=False,
                                   columns=[3], rows=[1], object_fit="contain",
                                   height="auto")

    # Generate
    gen_btn.click(img2img, inputs=[model, vae, taesd, img_inp,
                                   upscl, cnnet, control_img,
                                   control_strength, pprompt,
                                   nprompt, sampling, steps, schedule,
                                   width, height, batch_count,
                                   strenght, cfg, seed, clip_skip,
                                   threads, vae_tiling, cnnet_cpu, canny,
                                   rng, output, color, verbose],
                  outputs=[img_final])

    # Interactive Bindings
    reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                           outputs=[model])
    reload_vae_btn.click(reload_models, inputs=[vae_dir_txt], outputs=[vae])
    reload_taesd_btn.click(reload_models, inputs=[taesd_dir_txt],
                           outputs=[taesd])
    reload_upscl_btn.click(reload_models, inputs=[upscl_dir_txt],
                           outputs=[upscl])
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
                      outputs=[gallery, page_num_select, gallery])
    img2img_btn.click(reload_gallery, inputs=[img2img_ctrl],
                      outputs=[gallery, page_num_select, gallery])
    pvw_btn.click(prev_page, inputs=[], outputs=[gallery, page_num_select,
                                                 gallery])
    nxt_btn.click(next_page, inputs=[], outputs=[gallery, page_num_select,
                                                 gallery])
    first_btn.click(reload_gallery, inputs=[],
                    outputs=[gallery, page_num_select, gallery])
    last_btn.click(last_page, inputs=[],
                   outputs=[gallery, page_num_select, gallery])
    go_btn.click(goto_gallery, inputs=[page_num_select],
                 outputs=[gallery, page_num_select, gallery])


with gr.Blocks() as convert_block:
    # Title
    convert_title = gr.Markdown("# Convert and Quantize")

    with gr.Row():
        # Input
        with gr.Column(scale=1):
            with gr.Row():
                orig_model = gr.Dropdown(label="Original Model",
                                         choices=get_hf_models(), scale=5)
                reload_btn = gr.Button(RELOAD_SYMBOL, scale=1)
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
            reload_model_btn = gr.Button(value=RELOAD_SYMBOL, scale=1)
            reload_model_btn.click(reload_models, inputs=[model_dir_txt],
                                   outputs=[model])

        # VAE Dropdown
        vae = gr.Dropdown(label="VAE", choices=get_models(vae_dir), scale=7,
                          value=def_vae)
        with gr.Column(scale=1):
            reload_vae_btn = gr.Button(value=RELOAD_SYMBOL)
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
                               choices=schedulers,
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
            upscl_dir_txt = gr.Textbox(label="Upscaler folder", value=upscl_dir,
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
                                         upscl_dir_txt, cnnet_dir_txt,
                                         txt2img_dir_txt, img2img_dir_txt], [])
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
    main()

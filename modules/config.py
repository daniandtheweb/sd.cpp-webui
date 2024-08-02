"""sd.cpp-webui - Configuration module"""

import os
import json

CURRENT_DIR = os.getcwd()
CONFIG_PATH = 'config.json'

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
        'def_scheduler': in_schedule,
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
        'model_dir': 'os.path.join(CURRENT_DIR, "models/Stable-Diffusion/")',
        'vae_dir': 'os.path.join(CURRENT_DIR, "models/VAE/")',
        'emb_dir': 'os.path.join(CURRENT_DIR, "models/Embeddings/")',
        'lora_dir': 'os.path.join(CURRENT_DIR, "models/Lora/")',
        'taesd_dir': 'os.path.join(CURRENT_DIR, "models/TAESD/")',
        'upscl_dir' : 'os.path.join(CURRENT_DIR, "models/Upscalers/")',
        'cnnet_dir': 'os.path.join(CURRENT_DIR, "models/ControlNet/")',
        'txt2img_dir': 'os.path.join(CURRENT_DIR, "outputs/txt2img/")',
        'img2img_dir': 'os.path.join(CURRENT_DIR, "outputs/img2img/")',
        'def_sampling': "euler_a",
        'def_steps': 20,
        'def_scheduler': "discrete",
        'def_width': 512,
        'def_height': 512
    })

    data.pop('def_model', None)
    data.pop('def_vae', None)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(data, json_file_w, indent=4)

    print("Reset defaults completed.")


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
def_scheduler = data['def_scheduler']
def_width = data['def_width']
def_height = data['def_height']

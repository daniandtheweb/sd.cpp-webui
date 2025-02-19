"""sd.cpp-webui - Configuration module"""

import os
import json

import gradio as gr

CURRENT_DIR = os.getcwd()
CONFIG_PATH = 'config.json'
PROMPTS_PATH = 'prompts.json'


def set_defaults(in_ckpt, in_ckpt_vae, in_unet, in_unet_vae, in_clip_g,
                 in_clip_l, in_t5xxl, in_type, in_sampling, in_steps, in_schedule,
                 in_width, in_height, in_predict, in_ckpt_dir_txt,
                 in_unet_dir_txt, in_vae_dir_txt, in_clip_dir_txt,
                 in_emb_dir_txt, in_lora_dir_txt, in_taesd_dir_txt,
                 in_phtmkr_dir_txt, in_upscl_dir_txt, in_cnnet_dir_txt,
                 in_txt2img_dir_txt, in_img2img_dir_txt):
    """Sets new defaults"""
    # Directory defaults
    dir_defaults = {
        'ckpt_dir': in_ckpt_dir_txt,
        'vae_dir': in_vae_dir_txt,
        'unet_dir': in_unet_dir_txt,
        'clip_dir': in_clip_dir_txt,
        'emb_dir': in_emb_dir_txt,
        'lora_dir': in_lora_dir_txt,
        'taesd_dir': in_taesd_dir_txt,
        'phtmkr_dir': in_phtmkr_dir_txt,
        'upscl_dir': in_upscl_dir_txt,
        'cnnet_dir': in_cnnet_dir_txt,
        'txt2img_dir': in_txt2img_dir_txt,
        'img2img_dir': in_img2img_dir_txt,
    }
    data.update(dir_defaults)

    # Other defaults
    data.update({
        'def_type': in_type,
        'def_sampling': in_sampling,
        'def_steps': in_steps,
        'def_scheduler': in_schedule,
        'def_width': in_width,
        'def_height': in_height,
        'def_predict': in_predict
    })

    if in_ckpt:
        data['def_ckpt'] = in_ckpt
    if in_ckpt_vae:
        data['def_ckpt_vae'] = in_ckpt_vae
    if in_unet:
        data['def_unet'] = in_unet
    if in_unet_vae:
        data['def_unet_vae'] = in_unet_vae
    if in_clip_g:
        data['def_clip_g'] = in_clip_g
    if in_clip_l:
        data['def_clip_l'] = in_clip_l
    if in_t5xxl:
        data['def_t5xxl'] = in_t5xxl

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(data, json_file_w, indent=4)

    print("Set new defaults completed.")


def rst_def():
    """Restores factory defaults"""
    data.update({
        'ckpt_dir': os.path.join(CURRENT_DIR, "models/checkpoints/"),
        'unet_dir': os.path.join(CURRENT_DIR, "models/unet/"),
        'vae_dir': os.path.join(CURRENT_DIR, "models/vae/"),
        'clip_dir': os.path.join(CURRENT_DIR, "models/clip/"),
        'emb_dir': os.path.join(CURRENT_DIR, "models/embeddings/"),
        'lora_dir': os.path.join(CURRENT_DIR, "models/lora/"),
        'taesd_dir': os.path.join(CURRENT_DIR, "models/taesd/"),
        'phtmkr_dir': os.path.join(CURRENT_DIR, "models/photomaker/"),
        'upscl_dir': os.path.join(CURRENT_DIR, "models/upscale_models/"),
        'cnnet_dir': os.path.join(CURRENT_DIR, "models/controlnet/"),
        'txt2img_dir': os.path.join(CURRENT_DIR, "outputs/txt2img/"),
        'img2img_dir': os.path.join(CURRENT_DIR, "outputs/img2img/"),
        'def_type': "f16",
        'def_sampling': "euler_a",
        'def_steps': 20,
        'def_scheduler': "discrete",
        'def_width': 512,
        'def_height': 512,
        'def_predict': "Default"
    })

    data.pop('def_ckpt', None)
    data.pop('def_unet', None)
    data.pop('def_ckpt_vae', None)
    data.pop('def_unet_vae', None)
    data.pop('def_clip_l', None)
    data.pop('def_t5xxl', None)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(data, json_file_w, indent=4)

    print("Reset defaults completed.")


def get_prompts():
    """Lists saved prompts"""
    with open(PROMPTS_PATH, 'r', encoding="utf-8") as prompts_file:
        prompts_data = json.load(prompts_file)

    prompts_keys = list(prompts_data.keys())

    return prompts_keys


def reload_prompts():
    """Reloads prompts list"""
    refreshed_prompts = gr.update(choices=get_prompts())
    return refreshed_prompts


def save_prompts(prompt, pos_prompt, neg_prompt):
    """Saves a prompt"""
    if prompt is not None:
        with open(PROMPTS_PATH, 'r', encoding="utf-8") as prompts_file:
            prompts_data = json.load(prompts_file)

        prompts_data[prompt.strip()] = {
            'positive': pos_prompt,
            'negative': neg_prompt
        }

        with open(PROMPTS_PATH, 'w', encoding="utf-8") as prompts_file:
            json.dump(prompts_data, prompts_file, indent=4)
        print(f"Prompt '{prompt}' saved.")


def delete_prompts(prompt):
    """Deletes a saved prompt"""
    with open(PROMPTS_PATH, 'r', encoding="utf-8") as prompts_file:
        prompts_data = json.load(prompts_file)

    if prompt in prompts_data:
        del prompts_data[prompt]
        with open(PROMPTS_PATH, 'w', encoding="utf-8") as prompts_file:
            json.dump(prompts_data, prompts_file, indent=4)
        print(f"Prompt '{prompt}' deleted.")
    else:
        print(f"Prompt '{prompt}' not found.")


def load_prompts(prompt):
    """Loads a saved prompt"""
    with open(PROMPTS_PATH, 'r', encoding="utf-8") as prompts_file:
        prompts_data = json.load(prompts_file)
    key_data = prompts_data.get(prompt, {})
    pprompt_load = gr.update(value=key_data.get('positive', ''))
    nprompt_load = gr.update(value=key_data.get('negative', ''))
    return pprompt_load, nprompt_load


if not os.path.isfile(CONFIG_PATH):
    # Create an empty JSON file
    with open(CONFIG_PATH, 'w', encoding="utf-8") as config_file:
        # Write an empty JSON object
        json.dump({}, config_file, indent=4)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        data = json.load(config_file)
    rst_def()
    print("File 'config.json' created.")

with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    data = json.load(config_file)


ckpt_dir = data['ckpt_dir']
unet_dir = data['unet_dir']
vae_dir = data['vae_dir']
clip_dir = data['clip_dir']
emb_dir = data['emb_dir']
lora_dir = data['lora_dir']
taesd_dir = data['taesd_dir']
phtmkr_dir = data['phtmkr_dir']
upscl_dir = data['upscl_dir']
cnnet_dir = data['cnnet_dir']
txt2img_dir = data['txt2img_dir']
img2img_dir = data['img2img_dir']


if 'def_ckpt' in data:
    def_ckpt = data['def_ckpt']
else:
    def_ckpt = None
if 'def_ckpt_vae' in data:
    def_ckpt_vae = data['def_ckpt_vae']
else:
    def_ckpt_vae = None
if 'def_unet' in data:
    def_unet = data['def_unet']
else:
    def_unet = None
if 'def_unet_vae' in data:
    def_unet_vae = data['def_unet_vae']
else:
    def_unet_vae = None
if 'def_clip_g' in data:
    def_clip_g = data['def_clip_g']
else:
    def_clip_g = None
if 'def_clip_l' in data:
    def_clip_l = data['def_clip_l']
else:
    def_clip_l = None
if 'def_t5xxl' in data:
    def_t5xxl = data['def_t5xxl']
else:
    def_t5xxl = None
def_type = data['def_type']
def_sampling = data['def_sampling']
def_steps = data['def_steps']
def_scheduler = data['def_scheduler']
def_width = data['def_width']
def_height = data['def_height']
def_predict = data['def_predict']


if not os.path.isfile(PROMPTS_PATH):
    # Create an empty JSON file
    with open('prompts.json', 'w', encoding="utf-8") as prompts:
        # Write an empty JSON object
        json.dump({}, prompts, indent=4)
    print("File 'prompts.json' created.")

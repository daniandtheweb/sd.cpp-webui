"""sd.cpp-webui - Configuration module"""

import os
import json

import gradio as gr

CURRENT_DIR = os.getcwd()
CONFIG_PATH = 'config.json'
PROMPTS_PATH = 'prompts.json'


default_settings = {
    'ckpt_dir': os.path.join(CURRENT_DIR, "models/checkpoints/"),
    'unet_dir': os.path.join(CURRENT_DIR, "models/unet/"),
    'vae_dir': os.path.join(CURRENT_DIR, "models/vae/"),
    'clip_dir': os.path.join(CURRENT_DIR, "models/clip/"),
    'emb_dir': os.path.join(CURRENT_DIR, "models/embeddings/"),
    'lora_dir': os.path.join(CURRENT_DIR, "models/loras/"),
    'taesd_dir': os.path.join(CURRENT_DIR, "models/taesd/"),
    'phtmkr_dir': os.path.join(CURRENT_DIR, "models/photomaker/"),
    'upscl_dir': os.path.join(CURRENT_DIR, "models/upscale_models/"),
    'cnnet_dir': os.path.join(CURRENT_DIR, "models/controlnet/"),
    'txt2img_dir': os.path.join(CURRENT_DIR, "outputs/txt2img/"),
    'img2img_dir': os.path.join(CURRENT_DIR, "outputs/img2img/"),
    'def_type': "Default",
    'def_sampling': "euler_a",
    'def_steps': 20,
    'def_scheduler': "discrete",
    'def_width': 512,
    'def_height': 512,
    'def_predict': "Default",
    'def_flash_attn': False,
    'def_diffusion_conv_direct': False,
    'def_vae_conv_direct': False
}


def set_defaults(in_ckpt, in_ckpt_vae, in_unet, in_unet_vae, in_clip_g,
                 in_clip_l, in_t5xxl, in_type, in_sampling, in_steps, in_schedule,
                 in_width, in_height, in_predict, in_flash_attn,
                 in_diffusion_conv_direct, in_vae_conv_direct, in_ckpt_dir_txt,
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
    config_data.update(dir_defaults)

    # Other defaults
    config_data.update({
        'def_type': in_type,
        'def_sampling': in_sampling,
        'def_steps': in_steps,
        'def_scheduler': in_schedule,
        'def_width': in_width,
        'def_height': in_height,
        'def_predict': in_predict,
        'def_flash_attn': in_flash_attn,
        'def_diffusion_conv_direct': in_diffusion_conv_direct,
        'def_vae_conv_direct': in_vae_conv_direct
    })

    if in_ckpt:
        config_data['def_ckpt'] = in_ckpt
    if in_ckpt_vae:
        config_data['def_ckpt_vae'] = in_ckpt_vae
    if in_unet:
        config_data['def_unet'] = in_unet
    if in_unet_vae:
        config_data['def_unet_vae'] = in_unet_vae
    if in_clip_g:
        config_data['def_clip_g'] = in_clip_g
    if in_clip_l:
        config_data['def_clip_l'] = in_clip_l
    if in_t5xxl:
        config_data['def_t5xxl'] = in_t5xxl

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(config_data, json_file_w, indent=4)

    print("Set new defaults completed.")


def rst_def():
    """Restores factory defaults"""
    config_data.update(default_settings)

    config_data.pop('def_ckpt', None)
    config_data.pop('def_unet', None)
    config_data.pop('def_ckpt_vae', None)
    config_data.pop('def_unet_vae', None)
    config_data.pop('def_clip_l', None)
    config_data.pop('def_t5xxl', None)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as json_file_w:
        json.dump(config_data, json_file_w, indent=4)

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


# Load existing configuration or create an empty one
if os.path.isfile(CONFIG_PATH):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        config_data = json.load(config_file)
else:
    config_data = {}

# Update missing settings with defaults
updated = False
for key, value in default_settings.items():
    if key not in config_data:
        config_data[key] = value
        updated = True

# Save the updated configuration if changes were made
if updated:
    with open(CONFIG_PATH, 'w', encoding='utf-8') as config_file:
        json.dump(config_data, config_file, indent=4)
    print("Missing settings added to 'config.json'.")


ckpt_dir = config_data['ckpt_dir']
unet_dir = config_data['unet_dir']
vae_dir = config_data['vae_dir']
clip_dir = config_data['clip_dir']
emb_dir = config_data['emb_dir']
lora_dir = config_data['lora_dir']
taesd_dir = config_data['taesd_dir']
phtmkr_dir = config_data['phtmkr_dir']
upscl_dir = config_data['upscl_dir']
cnnet_dir = config_data['cnnet_dir']
txt2img_dir = config_data['txt2img_dir']
img2img_dir = config_data['img2img_dir']


if 'def_ckpt' in config_data:
    def_ckpt = config_data['def_ckpt']
else:
    def_ckpt = None
if 'def_ckpt_vae' in config_data:
    def_ckpt_vae = config_data['def_ckpt_vae']
else:
    def_ckpt_vae = None
if 'def_unet' in config_data:
    def_unet = config_data['def_unet']
else:
    def_unet = None
if 'def_unet_vae' in config_data:
    def_unet_vae = config_data['def_unet_vae']
else:
    def_unet_vae = None
if 'def_clip_g' in config_data:
    def_clip_g = config_data['def_clip_g']
else:
    def_clip_g = None
if 'def_clip_l' in config_data:
    def_clip_l = config_data['def_clip_l']
else:
    def_clip_l = None
if 'def_t5xxl' in config_data:
    def_t5xxl = config_data['def_t5xxl']
else:
    def_t5xxl = None
def_type = config_data['def_type']
def_sampling = config_data['def_sampling']
def_steps = config_data['def_steps']
def_scheduler = config_data['def_scheduler']
def_width = config_data['def_width']
def_height = config_data['def_height']
def_predict = config_data['def_predict']
def_flash_attn = config_data['def_flash_attn']
def_diffusion_conv_direct = config_data['def_diffusion_conv_direct']
def_vae_conv_direct = config_data['def_vae_conv_direct']


if not os.path.isfile(PROMPTS_PATH):
    # Create an empty JSON file
    with open('prompts.json', 'w', encoding="utf-8") as prompts_file:
        # Write an empty JSON object
        json.dump({}, prompts_file, indent=4)
    print("File 'prompts.json' created.")

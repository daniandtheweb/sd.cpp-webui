"""sd.cpp-webui - settings - what a fresh config starts with

DEFAULT_SETTINGS is the content of a new config.json:

- the folders ('*_dir'): where the models and the outputs live (below);
- the defaults ('def_<name>'): the saved starting value of each setting.
  They come from the option registry (modules/options/registry.py), the
  only place a default is written.
"""

import os

from modules.options import registry


CURRENT_DIR = os.getcwd()

# Folders: settings of the webui, not sd.cpp options
DIR_DEFAULTS = {
    'ckpt_dir': os.path.join(CURRENT_DIR, "models/checkpoints/"),
    'unet_dir': os.path.join(CURRENT_DIR, "models/unet/"),
    'vae_dir': os.path.join(CURRENT_DIR, "models/vae/"),
    'txt_enc_dir': os.path.join(CURRENT_DIR, "models/text_encoders/"),
    'tokenizer_dir': os.path.join(CURRENT_DIR, "models/tokenizers/"),
    'emb_dir': os.path.join(CURRENT_DIR, "models/embeddings/"),
    'lora_dir': os.path.join(CURRENT_DIR, "models/loras/"),
    'taesd_dir': os.path.join(CURRENT_DIR, "models/taesd/"),
    'phtmkr_dir': os.path.join(CURRENT_DIR, "models/photomaker/"),
    'upscl_dir': os.path.join(CURRENT_DIR, "models/upscale_models/"),
    'cnnet_dir': os.path.join(CURRENT_DIR, "models/controlnet/"),
    'ad_dir': os.path.join(CURRENT_DIR, "models/adetailer/"),
    'ipadapter_dir': os.path.join(CURRENT_DIR, "models/ipadapter/"),
    'pulid_dir': os.path.join(CURRENT_DIR, "models/pulid/"),
    'animatediff_dir': os.path.join(CURRENT_DIR, "models/animatediff/"),
    'txt2img_dir': os.path.join(CURRENT_DIR, "outputs/txt2img/"),
    'img2img_dir': os.path.join(CURRENT_DIR, "outputs/img2img/"),
    'imgedit_dir': os.path.join(CURRENT_DIR, "outputs/imgedit/"),
    'any2video_dir': os.path.join(CURRENT_DIR, "outputs/any2video/"),
    'upscale_dir': os.path.join(CURRENT_DIR, "outputs/upscale/")
}

DEFAULT_SETTINGS = {**DIR_DEFAULTS, **registry.config_defaults()}

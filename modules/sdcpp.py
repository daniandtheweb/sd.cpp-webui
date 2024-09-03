"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os

from modules.utility import run_subprocess, exe_name, get_path
from modules.gallery import get_next_img

from modules.config import (
    sd_dir, flux_dir, vae_dir, clip_l_dir, t5xxl_dir, emb_dir, lora_dir,
    taesd_dir, upscl_dir, cnnet_dir, txt2img_dir, img2img_dir
    )

SD = exe_name()


def txt2img(in_sd_model=None, in_sd_vae=None, in_flux_model=None,
            in_flux_vae=None, in_clip_l=None, in_t5xxl=None,
            in_model_type="Default", in_taesd=None, in_upscl=None,
            in_upscl_rep=1, in_cnnet=None, in_control_img=None,
            in_control_strength=1.0, in_ppromt="", in_nprompt="",
            in_sampling="default", in_steps=50, in_schedule="default",
            in_width=512, in_height=512, in_batch_count=1,
            in_cfg=7.0, in_seed=42, in_clip_skip=0, in_threads=1,
            in_vae_tiling=False, in_vae_cpu=False, in_cnnet_cpu=False,
            in_rng="default", in_predict="Default", in_output=None,
            in_color=False, in_verbose=False):

    """Text to image command creator"""
    fsd_model = get_path(sd_dir, in_sd_model)
    fsd_vae = get_path(vae_dir, in_sd_vae)
    fflux_model = get_path(flux_dir, in_flux_model)
    fflux_vae = get_path(vae_dir, in_flux_vae)
    fclip_l = get_path(clip_l_dir, in_clip_l)
    ft5xxl = get_path(t5xxl_dir, in_t5xxl)
    ftaesd = get_path(taesd_dir, in_taesd)
    fupscl = get_path(upscl_dir, in_upscl)
    fcnnet = get_path(cnnet_dir, in_cnnet)
    foutput = (os.path.join(txt2img_dir, f"{in_output}.png")
               if in_output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    # Initialize the command with prompts and critical options
    command = [SD, '-M', 'txt2img', '-p', f'"{in_ppromt}"']

    # Add prompts at the start
    if in_nprompt:
        command.extend(['-n', f'"{in_nprompt}"'])

    # Add image generation options
    command.extend([
        '--sampling-method', str(in_sampling),
        '--steps', str(in_steps),
        '--schedule', str(in_schedule),
        '-W', str(in_width),
        '-H', str(in_height),
        '-b', str(in_batch_count),
        '--cfg-scale', str(in_cfg),
        '-s', str(in_seed),
        '--clip-skip', str(in_clip_skip + 1),
        '--embd-dir', emb_dir,
        '--lora-model-dir', lora_dir,
        '-t', str(in_threads),
        '--rng', str(in_rng),
        '-o', foutput
    ])

    # Handle VAE options
    vae_option = fsd_vae if fsd_vae else fflux_vae

    # Optional parameters in dictionaries
    options = {
        # Model-related options
        '-m': fsd_model,
        '--diffusion-model': fflux_model,
        '--vae': vae_option,
        '--clip_l': fclip_l,
        '--t5xxl': ft5xxl,
        '--taesd': ftaesd,
        '--upscale-model': fupscl,
        '--upscale-repeats': str(in_upscl_rep) if fupscl else None,
        '--type': in_model_type if in_model_type != "Default" else None,
        # Control options
        '--control-net': fcnnet,
        '--control-image': in_control_img if fcnnet else None,
        '--control-strength': str(in_control_strength) if fcnnet else None,
        # Prediction mode
        '--prediction': in_predict if in_predict != "Default" else None
    }

    # Boolean flags
    flags = {
        '--vae-tiling': in_vae_tiling,
        '--vae-on-cpu': in_vae_cpu,
        '--control-net-cpu': in_cnnet_cpu,
        '--color': in_color,
        '-v': in_verbose
    }

    # Extend the command with options and their values
    for opt, value in options.items():
        if value is not None:
            command.extend([opt, value])

    # Add boolean flags
    for flag, condition in flags.items():
        if condition:
            command.append(flag)

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def img2img(in_sd, in_sd_vae, in_model_type, in_taesd, in_img_inp,
            in_upscl, in_upscl_rep, in_cnnet, in_control_img,
            in_control_strength, in_ppromt, in_nprompt, in_sampling,
            in_steps, in_schedule, in_width, in_height, in_batch_count,
            in_strenght, in_style_ratio, in_style_ratio_btn, in_cfg,
            in_seed, in_clip_skip, in_threads, in_vae_tiling, in_vae_cpu,
            in_cnnet_cpu, in_canny, in_rng, in_predict, in_output,
            in_color, in_verbose):
    """Image to image command creator"""
    fmodel = os.path.join(sd_dir, in_sd) if in_sd else None
    fsd_vae = os.path.join(vae_dir, in_sd_vae) if in_sd_vae else None
    ftaesd = os.path.join(taesd_dir, in_taesd) if in_taesd else None
    fupscl = os.path.join(upscl_dir, in_upscl) if in_upscl else None
    fcnnet = os.path.join(cnnet_dir, in_cnnet) if in_cnnet else None
    foutput = (os.path.join(img2img_dir, f"{in_output}.png")
               if in_output
               else os.path.join(img2img_dir, get_next_img(subctrl=1)))

    command = [SD, '-M', 'img2img', '-m', fmodel, '-i', in_img_inp, '-p',
               f'"{in_ppromt}"', '--sampling-method', str(in_sampling),
               '--steps', str(in_steps), '--schedule', f'{in_schedule}', '-W',
               str(in_width), '-H', str(in_height), '-b', str(in_batch_count),
               '--strength', str(in_strenght), '--cfg-scale', str(in_cfg),
               '-s', str(in_seed), '--clip-skip', str(in_clip_skip + 1),
               '--embd-dir', emb_dir, '--lora-model-dir', lora_dir, '-t',
               str(in_threads), '--rng', str(in_rng), '-o', foutput]

    if fsd_vae:
        command.extend(['--vae', fsd_vae])
    if str(in_model_type) != "Default":
        command.extend(['--type', str(in_model_type)])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
    if in_style_ratio_btn:
        command.extend(['--style-ratio', str(in_style_ratio)])
    if str(in_predict) != "Default":
        command.extend(['--prediction', str(in_predict)])
    if fupscl:
        command.extend(['--upscale-model', fupscl,
                        '--upscale-repeats', str(in_upscl_rep)])
    if fcnnet:
        command.extend(['--control-net', fcnnet, '--control-image',
                        in_control_img, '--control-strength',
                        str(in_control_strength)])
    if in_nprompt:
        command.extend(['-n', f'"{in_nprompt}"'])
    if in_vae_tiling:
        command.extend(['--vae-tiling'])
    if in_vae_cpu:
        command.extend(['--vae-on-cpu'])
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
    forig_model = os.path.join(sd_dir, in_orig_model)
    if not in_gguf_name:
        model_name, _ = os.path.splitext(in_orig_model)
        fgguf_name = f"{os.path.join(sd_dir, model_name)}"\
                     f".{in_quant_type}.gguf"
    else:
        fgguf_name = os.path.join(sd_dir, in_gguf_name)

    command = [SD, '-M', 'convert', '-m', forig_model,
               '-o', fgguf_name, '--type', in_quant_type]

    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return "Process completed."

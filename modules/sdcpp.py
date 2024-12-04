"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os

from modules.utility import subprocess_manager, exe_name, get_path
from modules.gallery import get_next_img
from modules.config import (
    sd_dir, flux_dir, vae_dir, clip_l_dir, t5xxl_dir, emb_dir, lora_dir,
    taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir, txt2img_dir, img2img_dir
)


SD = exe_name()


def txt2img(
    in_sd_model=None, in_sd_vae=None, in_flux_model=None,
    in_flux_vae=None, in_clip_l=None, in_t5xxl=None,
    in_model_type="Default", in_taesd=None, in_phtmkr=None,
    in_phtmkr_in=None, in_phtmkr_nrml=False, in_upscl=None,
    in_upscl_rep=1, in_cnnet=None, in_control_img=None,
    in_control_strength=1.0, in_ppromt="", in_nprompt="",
    in_sampling="default", in_steps=50, in_schedule="default",
    in_width=512, in_height=512, in_batch_count=1,
    in_cfg=7.0, in_seed=42, in_clip_skip=0, in_threads=1,
    in_vae_tiling=False, in_vae_cpu=False, in_cnnet_cpu=False,
    in_canny=False, in_rng="default", in_predict="Default",
    in_output=None, in_color=False, in_flash_attn=False,
    in_verbose=False
):

    """Text to image command creator"""
    fsd_model = get_path(sd_dir, in_sd_model)
    fsd_vae = get_path(vae_dir, in_sd_vae)
    fflux_model = get_path(flux_dir, in_flux_model)
    fflux_vae = get_path(vae_dir, in_flux_vae)
    fclip_l = get_path(clip_l_dir, in_clip_l)
    ft5xxl = get_path(t5xxl_dir, in_t5xxl)
    ftaesd = get_path(taesd_dir, in_taesd)
    fphtmkr = get_path(phtmkr_dir, in_phtmkr)
    fupscl = get_path(upscl_dir, in_upscl)
    fcnnet = get_path(cnnet_dir, in_cnnet)
    foutput = (os.path.join(txt2img_dir, f'{in_output}.png')
               if in_output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    # Initialize the command with prompts and critical options
    command = [SD, '-M', 'txt2img', '-p', f'{in_ppromt}']

    # Add prompts at the start
    if in_nprompt:
        command.extend(['-n', f'{in_nprompt}'])

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
        '--stacked-id-embd-dir': fphtmkr,
        '--input-id-images-dir': str(in_phtmkr_in) if fphtmkr else None,
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
        '--canny': in_canny,
        '--normalize-input': in_phtmkr_nrml,
        '--color': in_color,
        '--diffusion-fa': in_flash_attn,
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

    # Format prompts with brackets only for printing
    fppromt = f'"{in_ppromt}"'
    fnprompt = f'"{in_nprompt}"' if in_nprompt else ""

    # Replace prompts in the command for printing
    command_for_print = command.copy()

    # Find and replace the positive prompt in the command
    if '-p' in command_for_print:
        p_index = command_for_print.index('-p') + 1
        command_for_print[p_index] = fppromt

    # Find and replace the negative prompt in the command, if it exists
    if '-n' in command_for_print:
        n_index = command_for_print.index('-n') + 1
        command_for_print[n_index] = fnprompt

    # Construct the final command for printing
    fcommand = ' '.join(map(str, command_for_print))

    print(f"\n\n{fcommand}\n\n")
    subprocess_manager.run_subprocess(command)

    return [foutput]


def img2img(
    in_sd_model=None, in_sd_vae=None, in_flux_model=None,
    in_flux_vae=None, in_clip_l=None, in_t5xxl=None,
    in_model_type="Default", in_taesd=None, in_phtmkr=None,
    in_phtmkr_in=None, in_phtmkr_nrml=False, in_img_inp=None,
    in_upscl=None, in_upscl_rep=1, in_cnnet=None,
    in_control_img=None, in_control_strength=1.0, in_ppromt="",
    in_nprompt="", in_sampling="default", in_steps=50,
    in_schedule="default", in_width=512, in_height=512,
    in_batch_count=1, in_strenght=0.75, in_style_ratio=1.0,
    in_style_ratio_btn=False, in_cfg=7.0, in_seed=42, in_clip_skip=0,
    in_threads=1, in_vae_tiling=False, in_vae_cpu=False,
    in_cnnet_cpu=False, in_canny=False, in_rng="default",
    in_predict="Default", in_output=None, in_color=False,
    in_flash_attn=False, in_verbose=False
):

    """Image to image command creator"""
    # Construct file paths
    fsd_model = get_path(sd_dir, in_sd_model)
    fsd_vae = get_path(vae_dir, in_sd_vae)
    fflux_model = get_path(flux_dir, in_flux_model)
    fflux_vae = get_path(vae_dir, in_flux_vae)
    fclip_l = get_path(clip_l_dir, in_clip_l)
    ft5xxl = get_path(t5xxl_dir, in_t5xxl)
    ftaesd = get_path(taesd_dir, in_taesd)
    fphtmkr = get_path(phtmkr_dir, in_phtmkr)
    fupscl = get_path(upscl_dir, in_upscl)
    fcnnet = get_path(cnnet_dir, in_cnnet)
    foutput = (os.path.join(img2img_dir, f'{in_output}.png')
               if in_output
               else os.path.join(img2img_dir, get_next_img(subctrl=1)))

    # Initialize the command with prompts and critical options
    command = [SD, '-M', 'img2img', '-p', f'{in_ppromt}']

    # Add negative prompt if present
    if in_nprompt:
        command.extend(['-n', f'{in_nprompt}'])

    # Add image generation options
    command.extend([
        '-i', in_img_inp,
        '--sampling-method', str(in_sampling),
        '--steps', str(in_steps),
        '--schedule', str(in_schedule),
        '-W', str(in_width),
        '-H', str(in_height),
        '-b', str(in_batch_count),
        '--strength', str(in_strenght),
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
        '-m': fsd_model,
        '--diffusion-model': fflux_model,
        '--vae': vae_option,
        '--clip_l': fclip_l,
        '--t5xxl': ft5xxl,
        '--type': in_model_type if in_model_type != "Default" else None,
        '--taesd': ftaesd,
        '--stacked-id-embd-dir': fphtmkr,
        '--input-id-images-dir': str(in_phtmkr_in),
        '--style-ratio': str(in_style_ratio) if in_style_ratio_btn else None,
        '--prediction': in_predict if in_predict != "Default" else None,
        '--upscale-model': fupscl,
        '--upscale-repeats': str(in_upscl_rep) if fupscl else None,
        '--control-net': fcnnet,
        '--control-image': in_control_img if fcnnet else None,
        '--control-strength': str(in_control_strength) if fcnnet else None
    }

    # Boolean flags
    flags = {
        '--vae-tiling': in_vae_tiling,
        '--vae-on-cpu': in_vae_cpu,
        '--control-net-cpu': in_cnnet_cpu,
        '--normalize-input': in_phtmkr_nrml,
        '--canny': in_canny,
        '--color': in_color,
        '--diffusion-fa': in_flash_attn,
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

    # Format prompts with brackets only for printing
    fppromt = f'"{in_ppromt}"'
    fnprompt = f'"{in_nprompt}"' if in_nprompt else ""

    # Replace prompts in the command for printing
    command_for_print = command.copy()

    # Find and replace the positive prompt in the command
    if '-p' in command_for_print:
        p_index = command_for_print.index('-p') + 1
        command_for_print[p_index] = fppromt

    # Find and replace the negative prompt in the command, if it exists
    if '-n' in command_for_print:
        n_index = command_for_print.index('-n') + 1
        command_for_print[n_index] = fnprompt

    # Construct the final command for printing
    fcommand = ' '.join(map(str, command_for_print))

    print(f"\n\n{fcommand}\n\n")
    subprocess_manager.run_subprocess(command)

    return [foutput]


def convert(
    in_orig_model, in_model_dir, in_quant_type, in_gguf_name=None,
    in_verbose=False
):
    """Convert model command creator"""
    forig_model = os.path.join(in_model_dir, in_orig_model)
    if not in_gguf_name:
        model_name, _ = os.path.splitext(in_orig_model)
        model_path = os.path.join(in_model_dir, model_name)
        fgguf_name = f"{model_path}-{in_quant_type}.gguf"
    else:
        fgguf_name = os.path.join(in_model_dir, in_gguf_name)

    # Initialize base command
    command = [SD, '-M', 'convert']

    # Add essential command options
    command.extend([
        '-m', forig_model,        # Original model path
        '-o', fgguf_name,         # Output name
        '--type', in_quant_type   # Quantization type
    ])

    # Add verbosity flag if enabled
    if in_verbose:
        command.append('-v')

    # Convert command list to string format for printing
    fcommand = ' '.join(command)

    print(f"\n\n{fcommand}\n\n")
    subprocess_manager.run_subprocess(command)

    return "Process completed."

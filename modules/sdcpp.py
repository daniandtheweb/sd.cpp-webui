"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os

from modules.utility import subprocess_manager, exe_name, get_path
from modules.gallery import get_next_img
from modules.config import (
    ckpt_dir, unet_dir, vae_dir, clip_dir, emb_dir, lora_dir,
    taesd_dir, phtmkr_dir, upscl_dir, cnnet_dir, txt2img_dir, img2img_dir
)


SD = exe_name()


def command_generator(
    mode: str,
    prompt: str,
    nprompt: str,
    additional_args: list,
    options: dict,
    flags: dict,
    output_path: str,
    batch_count: int
) -> tuple:
    """
    Args:
        mode: e.g. 'img_gen'
        prompt: positive prompt
        nprompt: negative prompt
        additional_args: list of fixed arguments (e.g. sampling method, steps, width/height, etc.)
        options: dict of key-value pairs (only added if value is not None)
        flags: dict of boolean flags (only added if True)
        output_path: primary output file (including .png)
        batch_count: number of images

    Returns:
        (command_list, formatted_command_str, outputs_list)
    """
    # Start building the command list
    command = [SD, '-M', mode, '-p', prompt, '-n', nprompt] + additional_args

    # Add options with associated values
    for opt, val in options.items():
        if val is not None:
            command.extend([opt, val])
    # Add boolean flags
    for flag, condition in flags.items():
        if condition:
            command.append(flag)

    # Prepare a copy for printing: replace prompts with quoted versions
    command_for_print = command.copy()
    if '-p' in command_for_print:
        p_index = command_for_print.index('-p') + 1
        command_for_print[p_index] = f'"{prompt}"'
    if '-n' in command_for_print:
        n_index = command_for_print.index('-n') + 1
        command_for_print[n_index] = f'"{nprompt}"' if nprompt else ""
    fcommand = ' '.join(map(str, command_for_print))

    # Compute output filenames
    if batch_count == 1:
        outputs = [output_path]
    else:
        base = output_path[:-4]  # remove the ".png"
        outputs = [output_path] + [f"{base}_{i}.png" for i in range(2, batch_count + 1)]

    return command, fcommand, outputs


def txt2img(
    in_ckpt_model=None, in_ckpt_vae=None, in_unet_model=None,
    in_unet_vae=None, in_clip_g=None, in_clip_l=None, in_t5xxl=None,
    in_model_type="Default", in_taesd=None, in_phtmkr=None,
    in_phtmkr_in=None, in_phtmkr_nrml=False, in_upscl=None,
    in_upscl_rep=1, in_cnnet=None, in_control_img=None,
    in_control_strength=1.0, in_ppromt="", in_nprompt="",
    in_sampling="default", in_steps=50, in_scheduler="default",
    in_width=512, in_height=512, in_batch_count=1,
    in_cfg=7.0, in_seed=42, in_clip_skip=-1, in_threads=1,
    in_vae_tiling=False, in_vae_cpu=False, in_cnnet_cpu=False,
    in_canny=False, in_rng="default", in_predict="Default",
    in_output=None, in_color=False, in_flash_attn=False,
    in_diffusion_conv_direct=False, in_vae_conv_direct=False,
    in_verbose=False
):

    """Text to image command creator"""
    fckpt_model = get_path(ckpt_dir, in_ckpt_model)
    fckpt_vae = get_path(vae_dir, in_ckpt_vae)
    funet_model = get_path(unet_dir, in_unet_model)
    funet_vae = get_path(vae_dir, in_unet_vae)
    fclip_g = get_path(clip_dir, in_clip_g)
    fclip_l = get_path(clip_dir, in_clip_l)
    ft5xxl = get_path(clip_dir, in_t5xxl)
    ftaesd = get_path(taesd_dir, in_taesd)
    fphtmkr = get_path(phtmkr_dir, in_phtmkr)
    fupscl = get_path(upscl_dir, in_upscl)
    fcnnet = get_path(cnnet_dir, in_cnnet)
    foutput = (os.path.join(txt2img_dir, f'{in_output}.png')
               if in_output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    # Add image generation options
    additional_args = [
        '--sampling-method', str(in_sampling),
        '--steps', str(in_steps),
        '--scheduler', str(in_scheduler),
        '-W', str(in_width),
        '-H', str(in_height),
        '-b', str(in_batch_count),
        '--cfg-scale', str(in_cfg),
        '-s', str(in_seed),
        '--clip-skip', str(in_clip_skip),
        '--embd-dir', emb_dir,
        '--lora-model-dir', lora_dir,
        '-t', str(in_threads),
        '--rng', str(in_rng),
        '-o', foutput
    ]

    # Optional parameters in dictionaries
    options = {
        # Model-related options
        '--model': fckpt_model,
        '--diffusion-model': funet_model,
        '--vae': fckpt_vae if fckpt_vae else funet_vae,
        '--clip_g': fclip_g,
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
        '--diffusion-conv-direct': in_diffusion_conv_direct,
        '--vae-conv-direct': in_vae_conv_direct,
        '-v': in_verbose
    }

    command, fcommand, outputs = command_generator(
        mode="img_gen",
        prompt=in_ppromt,
        nprompt=in_nprompt,
        additional_args=additional_args,
        options=options,
        flags=flags,
        output_path=foutput,
        batch_count=in_batch_count
    )

    print(f"\n\n{fcommand}\n\n")
    yield fcommand, None

    subprocess_manager.run_subprocess(command)

    yield fcommand, outputs


def img2img(
    in_ckpt_model=None, in_ckpt_vae=None, in_unet_model=None,
    in_unet_vae=None, in_clip_g=None, in_clip_l=None, in_t5xxl=None,
    in_model_type="Default", in_taesd=None, in_phtmkr=None,
    in_phtmkr_in=None, in_phtmkr_nrml=False, in_img_inp=None,
    in_upscl=None, in_upscl_rep=1, in_cnnet=None,
    in_control_img=None, in_control_strength=1.0, in_ppromt="",
    in_nprompt="", in_sampling="default", in_steps=50,
    in_scheduler="default", in_width=512, in_height=512,
    in_batch_count=1, in_strenght=0.75, in_style_ratio=1.0,
    in_style_ratio_btn=False, in_cfg=7.0, in_seed=42, in_clip_skip=-1,
    in_threads=1, in_vae_tiling=False, in_vae_cpu=False,
    in_cnnet_cpu=False, in_canny=False, in_rng="default",
    in_predict="Default", in_output=None, in_color=False,
    in_flash_attn=False, in_diffusion_conv_direct=False,
    in_vae_conv_direct=False, in_verbose=False
):

    """Image to image command creator"""
    # Construct file paths
    fckpt_model = get_path(ckpt_dir, in_ckpt_model)
    fckpt_vae = get_path(vae_dir, in_ckpt_vae)
    funet_model = get_path(unet_dir, in_unet_model)
    funet_vae = get_path(vae_dir, in_unet_vae)
    fclip_g = get_path(clip_dir, in_clip_g)
    fclip_l = get_path(clip_dir, in_clip_l)
    ft5xxl = get_path(clip_dir, in_t5xxl)
    ftaesd = get_path(taesd_dir, in_taesd)
    fphtmkr = get_path(phtmkr_dir, in_phtmkr)
    fupscl = get_path(upscl_dir, in_upscl)
    fcnnet = get_path(cnnet_dir, in_cnnet)
    foutput = (os.path.join(img2img_dir, f'{in_output}.png')
               if in_output
               else os.path.join(img2img_dir, get_next_img(subctrl=1)))

    # Add image generation options
    additional_args = [
        '--init-img', str(in_img_inp),
        '--sampling-method', str(in_sampling),
        '--steps', str(in_steps),
        '--scheduler', str(in_scheduler),
        '-W', str(in_width),
        '-H', str(in_height),
        '-b', str(in_batch_count),
        '--strength', str(in_strenght),
        '--cfg-scale', str(in_cfg),
        '-s', str(in_seed),
        '--clip-skip', str(in_clip_skip),
        '--embd-dir', emb_dir,
        '--lora-model-dir', lora_dir,
        '-t', str(in_threads),
        '--rng', str(in_rng),
        '-o', foutput
    ]

    # Optional parameters in dictionaries
    options = {
        '--model': fckpt_model,
        '--diffusion-model': funet_model,
        '--vae': fckpt_vae if fckpt_vae else funet_vae,
        '--clip_g': fclip_g,
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
        '--diffusion-conv-direct': in_diffusion_conv_direct,
        '--vae-conv-direct': in_vae_conv_direct,
        '-v': in_verbose
    }

    command, fcommand, outputs = command_generator(
        mode="img_gen",
        prompt=in_ppromt,
        nprompt=in_nprompt,
        additional_args=additional_args,
        options=options,
        flags=flags,
        output_path=foutput,
        batch_count=in_batch_count
    )

    print(f"\n\n{fcommand}\n\n")
    yield fcommand, None

    subprocess_manager.run_subprocess(command)

    yield fcommand, outputs


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
        '--model', forig_model,   # Original model path
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

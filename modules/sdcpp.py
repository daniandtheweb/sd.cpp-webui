"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os

from modules.utility import run_subprocess, exe_name
from modules.gallery import get_next_img

from modules.config import (
    model_dir, vae_dir,emb_dir, lora_dir,taesd_dir, upscl_dir,
    cnnet_dir, txt2img_dir, img2img_dir
    )

SD = exe_name()

def txt2img(in_model, in_vae, in_taesd, in_upscl, in_upscl_rep, in_cnnet, in_control_img,
            in_control_strength, in_ppromt, in_nprompt, in_sampling,
            in_steps, in_schedule, in_width, in_height, in_batch_count,
            in_cfg, in_seed, in_clip_skip, in_threads, in_vae_tiling,
            in_vae_cpu, in_cnnet_cpu, in_rng, in_output, in_color,
            in_verbose):
    """Text to image command creator"""
    fmodel = os.path.join(model_dir, in_model) if in_model else None
    fvae = os.path.join(vae_dir, in_vae) if in_vae else None
    ftaesd = os.path.join(taesd_dir, in_taesd) if in_taesd else None
    fupscl = os.path.join(upscl_dir, in_upscl) if in_upscl else None
    fcnnet = os.path.join(cnnet_dir, in_cnnet) if in_cnnet else None
    foutput = (os.path.join(txt2img_dir, f"{in_output}.png")
               if in_output
               else os.path.join(txt2img_dir, get_next_img(subctrl=0)))

    command = [SD, '-M', 'txt2img', '-m', fmodel, '-p', f'"{in_ppromt}"',
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
    if in_color:
        command.extend(['--color'])
    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return [foutput]


def img2img(in_model, in_vae, in_taesd, in_img_inp, in_upscl, in_upscl_rep, in_cnnet,
            in_control_img, in_control_strength, in_ppromt, in_nprompt,
            in_sampling, in_steps, in_schedule, in_width, in_height,
            in_batch_count, in_strenght, in_cfg, in_seed, in_clip_skip,
            in_threads, in_vae_tiling, in_vae_cpu, in_cnnet_cpu, in_canny,
            in_rng, in_output, in_color, in_verbose):
    """Image to image command creator"""
    fmodel = os.path.join(model_dir, in_model) if in_model else None
    fvae = os.path.join(vae_dir, in_vae) if in_vae else None
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

    if fvae:
        command.extend(['--vae', fvae])
    if ftaesd:
        command.extend(['--taesd', ftaesd])
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
    forig_model = os.path.join(model_dir, in_orig_model)
    if not in_gguf_name:
        model_name, _ = os.path.splitext(in_orig_model)
        fgguf_name = f"{os.path.join(model_dir, model_name)}"\
                     f".{in_quant_type}.gguf"
    else:
        fgguf_name = os.path.join(model_dir, in_gguf_name)

    command = [SD, '-M', 'convert', '-m', forig_model,
               '-o', fgguf_name, '--type', in_quant_type]

    if in_verbose:
        command.extend(['-v'])

    fcommand = ' '.join(map(str, command))

    print(fcommand)
    run_subprocess(command)

    return "Process completed."

"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os
from enum import IntEnum
from typing import Dict, Any, Generator

import gradio as gr

from modules.utils.utility import get_path
from modules.gallery import get_next_img
from modules.shared_instance import (
    config, subprocess_manager, SD
)


class DiffusionMode(IntEnum):
    CHECKPOINT = 0
    UNET = 1


class CommandRunner:
    """Builds and runs stable-diffusion.cpp commands and yelds UI updates."""

    def __init__(self, mode: str, params: Dict[str, Any]):
        self.mode = mode
        self.params = params
        self.env_vars = self._extract_env_vars()
        self.command = [SD, '-M', self.mode]
        self.fcommand = ""
        self.outputs = []
        self.output_path = ""
        self.preview_path = None

    def _extract_env_vars(self) -> Dict[str, Any]:
        """
        Parses the params dictionary to find and extract environment
        variables, applying conditional logic as needed.
        """
        env_vars = {}

        is_vk_override_true = self.params.pop('env_vk_visible_override', False)
        vk_device_id = self.params.pop('env_GGML_VK_VISIBLE_DEVICES', None)
        is_cuda_override_true = self.params.pop('env_cuda_visible_override', False)
        cuda_device_id = self.params.pop('env_CUDA_VISIBLE_DEVICES', None)

        for key in list(self.params.keys()):
            if key.startswith("env_"):
                env_key = key[4:]
                value = self.params.pop(key)
                if env_key not in env_vars:
                    env_vars[env_key] = value

        if is_vk_override_true and vk_device_id is not None:
            env_vars['GGML_VK_VISIBLE_DEVICES'] = vk_device_id
        if is_cuda_override_true and cuda_device_id is not None:
            env_vars['CUDA_VISIBLE_DEVICES'] = cuda_device_id

        return env_vars

    def _set_output_path(self, dir_key: str, subctrl_id: int, extension: str):
        """Determines and sets the output path for the command."""
        output_dir = config.get(dir_key)
        filename = self._get_param('in_output')

        if filename:
            self.output_path = os.path.join(output_dir, f"{filename}.{extension}")
        else:
            self.output_path = os.path.join(output_dir, get_next_img(subctrl=subctrl_id))

    def _resolve_paths(self):
        """Resolves all model and directory paths from the config."""
        path_mappings = {
            'ckpt_dir': ['in_ckpt_model'],
            'vae_dir': ['in_ckpt_vae', 'in_unet_vae'],
            'unet_dir': ['in_unet_model', 'in_high_noise_model'],
            'clip_dir': [
                'in_clip_g', 'in_clip_l', 'in_t5xxl', 'in_qwen2vl',
                'in_umt5_xxl', 'in_clip_vision_h'
            ],
            'taesd_dir': ['in_taesd'],
            'phtmkr_dir': ['in_phtmkr'],
            'upscl_dir': ['in_upscl'],
            'cnnet_dir': ['in_cnnet']
        }
        for dir_key, param_keys in path_mappings.items():
            for param_key in param_keys:
                if param_key in self.params:
                    # Create a new key for the full path, e.g., 'f_ckpt_model'
                    full_path_key = f"f_{param_key.replace('in_', '')}"
                    self.params[full_path_key] = get_path(
                        config.get(dir_key), self.params.get(param_key)
                    )

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Helper to get a parameter from the params dictionary."""
        return self.params.get(key, default)

    def _add_base_args(self):
        """Adds arguments common to all modes."""
        self.command.extend([
            '--sampling-method', str(self._get_param('in_sampling')),
            '--steps', str(self._get_param('in_steps')),
            '--scheduler', str(self._get_param('in_scheduler')),
            '-W', str(self._get_param('in_width')),
            '-H', str(self._get_param('in_height')),
            '-b', str(self._get_param('in_batch_count')),
            '--cfg-scale', str(self._get_param('in_cfg')),
            '-s', str(self._get_param('in_seed')),
            '--clip-skip', str(self._get_param('in_clip_skip')),
            '--embd-dir', config.get('emb_dir'),
            '--lora-model-dir', config.get('lora_dir'),
            '-t', str(self._get_param('in_threads')),
            '--rng', str(self._get_param('in_rng')),
            '-o', self.output_path
        ])

    def _add_options(self, options: Dict[str, Any]):
        """Adds key-value options to the command if the value is not None."""
        for opt, val in options.items():
            if val is not None:
                self.command.extend([opt, str(val)])

    def _add_flags(self, flags: Dict[str, bool]):
        """Adds boolean flags to the command if they are True."""
        for flag, condition in flags.items():
            if condition:
                self.command.append(flag)

    def _prepare_for_run(self):
        """
        Prepares the final command string for printing and computes outputs.
        """
        prompt = self._get_param('in_pprompt', "")
        nprompt = self._get_param('in_nprompt', "")

        # Prepare a copy for printing
        cmd_print = self.command.copy()
        if '-p' in cmd_print:
            cmd_print[cmd_print.index('-p') + 1] = f'"{prompt}"'
        if '-n' in cmd_print:
            cmd_print[cmd_print.index('-n') + 1] = f'"{nprompt}"'
        self.fcommand = ' '.join(map(str, cmd_print))

        # Compute all output filenames
        batch_count = self._get_param('in_batch_count', 1)
        if batch_count == 1:
            self.outputs = [self.output_path]
        else:
            base, ext = os.path.splitext(self.output_path)
            self.outputs = [self.output_path] + [
                f"{base}_{i}{ext}" for i in range(2, batch_count + 1)
            ]

    def build_command(self):
        """
        Main method to build the command.
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def run(self) -> Generator:
        """Runs the command and yields Gradio updates."""
        self._prepare_for_run()
        print(f"\n\n{self.fcommand}\n\n")

        process_env = os.environ.copy()

        if self.env_vars:
            settings_to_print = []

            for key, value in self.env_vars.items():
                if isinstance(value, bool):
                    if value is True:
                        process_env[key] = "1"
                        settings_to_print.append(f"{key}=1")
                elif isinstance(value, int):
                    process_env[key] = str(value)
                    settings_to_print.append(f"{key}={str(value)}")
            if settings_to_print:
                full_line = " ".join(settings_to_print)
                print(f"  SET: {full_line}\n\n")

        if self.preview_path:
            gallery_update = [self.preview_path]
        else:
            gallery_update = None

        yield (self.fcommand, gr.update(visible=True, value=0),
               gr.update(visible=True, value="Initializing..."),
               gr.update(value=""), None)

        final_stats_str = "Process completed with unknown stats."
        for update in subprocess_manager.run_subprocess(
            self.command, env=process_env
        ):
            if "final_stats" in update:
                stats = update["final_stats"]
                final_stats_str = (
                    f"Tensor Load: {stats.get('tensor_load_time', 'N/A')} | "
                    f"Sampling: {stats.get('sampling_time', 'N/A')} | "
                    f"Decode: {stats.get('decoding_time', 'N/A')} | "
                    f"Total: {stats.get('total_time', 'N/A')} | "
                    f"Last Speed: {stats.get('last_speed', 'N/A')}"
                )
            else:
                yield (self.fcommand, gr.update(value=update["percent"]),
                       update["status"], gr.update(value=""), gallery_update)

        if self.preview_path:
            os.remove(self.preview_path)

        yield (self.fcommand, gr.update(visible=False, value=100),
               gr.update(visible=False, value=""),
               gr.update(value=final_stats_str), self.outputs)


class ImageGenerationRunner(CommandRunner):
    """A common base for txt2img, img2img, and imgedit runners."""

    def _get_model_options(self) -> Dict[str, Any]:
        """Builds and returns a dictionary of model and VAE options."""
        options = {}
        diffusion_mode = self._get_param('in_diffusion_mode')

        if diffusion_mode == DiffusionMode.CHECKPOINT:
            options['--model'] = self._get_param('f_ckpt_model')
            options['--vae'] = self._get_param('f_ckpt_vae')
        elif diffusion_mode == DiffusionMode.UNET:
            options['--diffusion-model'] = self._get_param('f_unet_model')
            options['--vae'] = self._get_param('f_unet_vae')
            options['--clip_g'] = self._get_param('f_clip_g')
            options['--clip_l'] = self._get_param('f_clip_l')
            options['--t5xxl'] = self._get_param('f_t5xxl')
            options['--qwen2vl'] = self._get_param('f_qwen2vl')
            options['--qwen2vl_vision'] = self._get_param('f_qwen2vl_vision')

        # Filter out any keys that have a None value before returning
        return {k: v for k, v in options.items() if v is not None}


    def build_command(self, output_dir_key: str, subctrl_id: int):
        """Builds the common command for image generation."""
        self._resolve_paths()
        self._set_output_path(output_dir_key, subctrl_id, 'png')

        self.command.extend(['-p', self._get_param('in_pprompt', "")])
        if self._get_param('in_nprompt'):
            self.command.extend(['-n', self._get_param('in_nprompt')])

        self._add_base_args()

        if self._get_param('in_preview_bool'):
            self.preview_path = self.output_path + "preview.png"

        options = {
            # Models
            **self._get_model_options(),
            # Weight type
            '--type': (self._get_param('in_model_type')
                       if self._get_param('in_model_type') != "Default"
                       else None),
            '--tensor-type-rules': (self._get_param('in_tensor_type_rules')
                                    if self._get_param('in_tensor_type_rules') != ""
                                    else None),
            # TAESD
            '--taesd': self._get_param('f_taesd'),
            # PhotoMaker
            **({
                '--photo-maker': self._get_param('f_phtmkr'),
                '--pm-id-images-dir': self._get_param('in_phtmkr_id'),
                '--pm-id-embed-path': self._get_param('in_phtmkr_emb'),
                '--pm-style-strength': self._get_param('in_phtmkr_strength')
            } if self._get_param('in_phtmkr_bool') else {}),
            # Guidance
            '--guidance': (self._get_param('in_guidance')
                           if self._get_param('in_guidance_bool')
                           else None),
            # Flow Shift
            '--flow-shift': (self._get_param('in_flow_shift')
                             if self._get_param('in_flow_shift_bool')
                             else None),
            # Timestep shift for NitroFusion
            '--timestep-shift': (self._get_param('in_timestep_shift')
                                 if self._get_param('in_timestep_shift_bool')
                                 else None),
            # ETA for DDIM and TCD
            '--eta': (self._get_param('in_eta')
                      if self._get_param('in_eta_bool')
                      else None),
            # Upscale
            **({
                '--upscale-model': self._get_param('f_upscl'),
                '--upscale-repeats': self._get_param('in_upscl_rep')
            } if self._get_param('in_upscl_bool') else {}),
            # ControlNet
            **({
                '--control-net': self._get_param('f_cnnet'),
                '--control-image': self._get_param('in_control_img'),
                '--control-strength': self._get_param('in_control_strength')
            } if self._get_param('in_cnnet_bool') else {}),
            # Chroma
            '--chroma-t5-mask-pad': (self._get_param('in_t5_mask_pad')
                                     if self._get_param('in_enable_t5_mask')
                                     else None),
            # VAE Tiling
            **({
                '--vae-tile-overlap': self._get_param('in_vae_tile_overlap'),
                '--vae-tile-size': (f"{size}x{size}"
                                    if (size := self._get_param('in_vae_tile_size'))
                                    else None),
                '--vae-relative-tile-size': (f"{size}x{size}"
                                             if (
                                             self._get_param('in_vae_relative_bool') and
                                             (size := self._get_param('in_vae_relative_tile_size'))
                                             )
                                             else None),
            } if self._get_param('in_vae_tiling') else {}),
            # Prediction type override
            '--prediction': (self._get_param('in_predict')
                             if self._get_param('in_predict') != "Default"
                             else None),
            # Preview
            **({
                '--preview': self._get_param('in_preview_mode'),
                '--preview-path': self.preview_path,
                '--preview-interval': self._get_param('in_preview_interval'),
            } if self._get_param('in_preview_bool') else {})
        }
        self._add_options(options)

        flags = {
            '--offload-to-cpu': self._get_param('in_offload_to_cpu'),
            '--vae-tiling': self._get_param('in_vae_tiling'),
            '--vae-on-cpu': self._get_param('in_vae_cpu'),
            '--clip-on-cpu': self._get_param('in_clip_cpu'),
            '--control-net-cpu': self._get_param('in_cnnet_cpu'),
            '--canny': self._get_param('in_canny'),
            '--chroma-disable-dit-mask': (
                self._get_param('in_disable_dit_mask')
            ),
            '--chroma-enable-t5-mask': self._get_param('in_enable_t5_mask'),
            '--diffusion-fa': self._get_param('in_flash_attn'),
            '--diffusion-conv-direct': (
                self._get_param('in_diffusion_conv_direct')
            ),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '--force-sdxl-vae-conv-scale': self._get_param('in_force_sdxl_vae_conv_scale'),
            '--taesd-preview-only': (self._get_param('in_preview_taesd')
                                     if self._get_param('in_preview_bool')
                                     else False),
            '--preview-noisy': (self._get_param('in_preview_noisy')
                                if self._get_param('in_preview_bool')
                                else False),
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose'),
        }
        self._add_flags(flags)


class Txt2ImgRunner(ImageGenerationRunner):
    """Builds the txt2img command."""
    def build_command(self):
        super().build_command(
            output_dir_key='txt2img_dir', subctrl_id=0
        )


class Img2ImgRunner(ImageGenerationRunner):
    """Builds the img2img command."""
    def build_command(self):
        super().build_command(
            output_dir_key='img2img_dir', subctrl_id=1
        )

        # Add img2img specific arguments
        self.command.extend(['--init-img', str(self._get_param('in_img_inp'))])
        self.command.extend([
            '--strength',
            str(self._get_param('in_strength'))
        ])

        options = {
            '--img-cfg-scale': (self._get_param('in_img_cfg')
                                if self._get_param('in_img_cfg_bool')
                                else None),
        }
        self._add_options(options)


class ImgEditRunner(ImageGenerationRunner):
    """Builds the image editing (instruct) command."""
    def build_command(self):
        super().build_command(
            output_dir_key='imgedit_dir', subctrl_id=2
        )
        self.command.extend(['--ref-image', str(self._get_param('in_ref_img'))])


class Any2VideoRunner(CommandRunner):
    """Builds the any2video command."""
    def _add_base_args(self):
        # Override to add video-specific base arguments
        super()._add_base_args()
        self.command.extend([
            '--video-frames', str(self._get_param('in_frames')),
            '--fps', str(self._get_param('in_fps')),
        ])

    def build_command(self):
        self._resolve_paths()
        self._set_output_path('any2video_dir', 3, 'avi')

        self.command.extend(['-p', self._get_param('in_pprompt', "")])
        if self._get_param('in_nprompt'):
            self.command.extend(['-n', self._get_param('in_nprompt')])

        self._add_base_args()

        init_img = (self._get_param('in_img_inp')
                    or self._get_param('in_first_frame_inp'))
        options = {
            # VAE
            '--vae': self._get_param('f_unet_vae'),
            # Wan2.1, Wan2.2
            '--diffusion-model': self._get_param('f_unet_model'),
            '--clip_vision': self._get_param('f_clip_vision_h'),
            '--t5xxl': self._get_param('f_umt5_xxl'),
            '--high-noise-diffusion-model': (
                self._get_param('f_high_noise_model')
            ),
            # TAESD
            '--taesd': self._get_param('f_taesd'),
            '--init-img': init_img,
            '--end-img': self._get_param('in_last_frame_inp'),
            '--upscale-model': (self._get_param('f_upscl')
                                if self._get_param('in_upscl_bool')
                                else None),
            '--upscale-repeats': (self._get_param('in_upscl_rep')
                                  if self._get_param('in_upscl_bool')
                                  else None),
            '--type': (self._get_param('in_model_type')
                       if self._get_param('in_model_type') != "Default"
                       else None),
            '--flow-shift': (self._get_param('in_flow_shift')
                             if self._get_param('in_flow_shift_bool')
                             else None),
            # ControlNet
            '--control-net': (self._get_param('f_cnnet')
                              if self._get_param('in_cnnet_bool')
                              else None),
            '--control-image': (self._get_param('in_control_img')
                                if self._get_param('in_cnnet_bool')
                                else None),
            '--control-strength': (self._get_param('in_control_strength')
                                   if self._get_param('in_cnnet_bool')
                                   else None),
            '--prediction': (self._get_param('in_predict')
                             if self._get_param('in_predict') != "Default"
                             else None)
        }
        self._add_options(options)

        flags = {
            '--offload-to-cpu': self._get_param('in_offload_to_cpu'),
            '--vae-tiling': self._get_param('in_vae_tiling'),
            '--vae-on-cpu': self._get_param('in_vae_cpu'),
            '--clip-on-cpu': self._get_param('in_clip_cpu'),
            '--control-net-cpu': self._get_param('in_cnnet_cpu'),
            '--canny': self._get_param('in_canny'),
            '--color': self._get_param('in_color'),
            '--diffusion-fa': self._get_param('in_flash_attn'),
            '--diffusion-conv-direct': (
                self._get_param('in_diffusion_conv_direct')
            ),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '-v': self._get_param('in_verbose')
        }
        self._add_flags(flags)


class UpscaleRunner(CommandRunner):
    """Builds the upscale command."""
    def build_command(self):
        self._resolve_paths()
        self._set_output_path('upscale_dir', 4, 'png')

        init_img = (self._get_param('in_img_inp')
                    or self._get_param('in_first_frame_inp'))
        options = {
            '--init-img': init_img,
            '--upscale-model': self._get_param('f_upscl'),
            '-W': str(self._get_param('in_init_width')),
            '-H': str(self._get_param('in_init_height')),
            '--upscale-repeats': self._get_param('in_upscl_rep'),
            '-o': self.output_path,
        }
        self._add_options(options)

        flags = {
            '--diffusion-fa': self._get_param('in_flash_attn'),
            '--diffusion-conv-direct': (
                self._get_param('in_diffusion_conv_direct')
            ),
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose')
        }
        self._add_flags(flags)


def txt2img(params: dict) -> Generator:
    """Creates and runs a Txt2ImgRunner."""
    runner = Txt2ImgRunner(mode="img_gen", params=params)
    runner.build_command()
    yield from runner.run()


def img2img(params: dict) -> Generator:
    """Creates and runs an Img2ImgRunner."""
    runner = Img2ImgRunner(mode="img_gen", params=params)
    runner.build_command()
    yield from runner.run()


def imgedit(params: dict) -> Generator:
    """Creates and runs an ImgEditRunner."""
    runner = ImgEditRunner(mode="img_gen", params=params)
    runner.build_command()
    yield from runner.run()


def any2video(params: dict) -> Generator:
    """Creates and runs an Any2VideoRunner."""
    runner = Any2VideoRunner(mode="vid_gen", params=params)
    runner.build_command()
    yield from runner.run()

def upscale(params: dict) -> Generator:
    """Creates and runs an UpscaleRunner."""
    runner = UpscaleRunner(mode="upscale", params=params)
    runner.build_command()
    yield from runner.run()

def convert(
    in_orig_model: str, in_model_dir: str, in_quant_type: str, in_tensor_type_rules: str = None,
    in_gguf_name: str = None, in_verbose: bool = False
) -> str:
    """Synchronously runs the model conversion command."""
    orig_model_path = os.path.join(in_model_dir, in_orig_model)

    if in_gguf_name:
        gguf_path = os.path.join(in_model_dir, in_gguf_name)
    else:
        model_name, _ = os.path.splitext(in_orig_model)
        gguf_path = os.path.join(
            in_model_dir, f"{model_name}-{in_quant_type}.gguf"
        )

    command = [
        SD, '-M', 'convert',
        '--model', orig_model_path,
        '-o', gguf_path,
        '--type', in_quant_type
    ]
    if in_tensor_type_rules:
        command.extend(['--tensor-type-rules', in_tensor_type_rules])
    if in_verbose:
        command.append('-v')

    fcommand = ' '.join(command)
    print(f"\n\n{fcommand}\n\n")

    # This assumes run_subprocess can handle synchronous execution
    # and will block until the process is complete.
    for _ in subprocess_manager.run_subprocess(command):
        pass  # Consume generator if it's async

    return "Process completed."

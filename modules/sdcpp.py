"""sd.cpp-webui - stable-diffusion.cpp command module"""

import os
from typing import Dict, Any, Generator

import gradio as gr

from modules.utils.utility import get_path
from modules.gallery import get_next_img
from modules.shared_instance import (
    config, subprocess_manager, SD
)


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


class Txt2ImgRunner(CommandRunner):
    def build_command(self):
        self._resolve_paths()
        self.output_path = (
            os.path.join(
                config.get('txt2img_dir'),
                f"{self._get_param('in_output')}.png"
            )
            if self._get_param('in_output')
            else os.path.join(
                config.get('txt2img_dir'), get_next_img(subctrl=0)
            )
        )

        self.command.extend(['-p', self._get_param('in_pprompt', "")])
        if self._get_param('in_nprompt'):
            self.command.extend(['-n', self._get_param('in_nprompt')])

        self._add_base_args()

        preview_mode = self._get_param('in_preview_mode')
        is_preview_enabled = (
            preview_mode
            if preview_mode is not None
            and preview_mode != "none"
            else None
        )
        if is_preview_enabled:
            self.preview_path = self.output_path + "preview.png"

        options = {
            # Weight type
            '--type': (self._get_param('in_model_type')
                       if self._get_param('in_model_type') != "Default"
                       else None),
            # VAE
            '--vae': (self._get_param('f_ckpt_vae')
                      if self._get_param('in_diffusion_mode') == 0
                      else self._get_param('f_unet_vae')
                      if self._get_param('in_diffusion_mode') == 1
                      else None),
            # SD1.x, SD2.x, SD-Turbo, SDXL, SDXL-Turbo, NitroFusion
            '--model': (self._get_param('f_ckpt_model')
                        if self._get_param('in_diffusion_mode') == 0
                        else None),
            # SD3, SD3.5, Flux-dev, Flux-schnell, Chroma, Qwen Image
            '--diffusion-model': (self._get_param('f_unet_model')
                                  if self._get_param('in_diffusion_mode') == 1
                                  else None),
            '--clip_g': (self._get_param('f_clip_g')
                         if self._get_param('in_diffusion_mode') == 1
                         else None),
            '--clip_l': (self._get_param('f_clip_l')
                         if self._get_param('in_diffusion_mode') == 1
                         else None),
            '--t5xxl': (self._get_param('f_t5xxl')
                        if self._get_param('in_diffusion_mode') == 1
                        else None),
            '--qwen2vl': (self._get_param('f_qwen2vl')
                          if self._get_param('in_diffusion_mode') == 1
                          else None),
            # TAESD
            '--taesd': self._get_param('f_taesd'),
            # PhotoMaker
            '--photo-maker': (self._get_param('f_phtmkr')
                              if self._get_param('in_phtmkr_bool')
                              else None),
            '--pm-id-images-dir': (self._get_param('f_phtmkr_id')
                                   if self._get_param('in_phtmkr_bool')
                                   else None),
            '--pm-id-embed-path': (self._get_param('f_phtmkr_emb')
                                      if self._get_param('in_phtmkr_bool')
                                      else None),
            '--pm-style-strength': (self._get_param('in_phtmkr_strength')
                                    if self._get_param('in_phtmkr_bool')
                                    else None),
            '--guidance': (self._get_param('in_guidance')
                           if self._get_param('in_guidance_bool')
                           else None),
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
            '--upscale-model': (self._get_param('f_upscl')
                                if self._get_param('in_upscl_bool')
                                else None),
            '--upscale-repeats': (self._get_param('in_upscl_rep')
                                  if self._get_param('in_upscl_bool')
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
            '--chroma-t5-mask-pad': (self._get_param('in_t5_mask_pad')
                                     if self._get_param('in_enable_t5_mask')
                                     else None
                                     ),
            # Prediction type override
            '--prediction': (self._get_param('in_predict')
                             if self._get_param('in_predict') != "Default"
                             else None),
            # Preview
            '--preview': (self._get_param('in_preview_mode')
                          if is_preview_enabled
                          else None),
            '--preview-path': (self.preview_path
                               if is_preview_enabled
                               else None),
            '--preview-interval': (self._get_param('in_preview_interval')
                                   if is_preview_enabled
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
            '--chroma-disable-dit-mask': (
                self._get_param('in_disable_dit_mask')
            ),
            '--chroma-enable-t5-mask': self._get_param('in_enable_t5_mask'),
            '--color': self._get_param('in_color'),
            '--diffusion-fa': self._get_param('in_flash_attn'),
            '--diffusion-conv-direct': (
                self._get_param('in_diffusion_conv_direct')
            ),
            '--vae-conv-direct': self._get_param('in_vae_conv_direct'),
            '-v': self._get_param('in_verbose'),
            '--taesd-preview-only': (self._get_param('in_preview_taesd')
                                     if is_preview_enabled
                                     else False)
        }
        self._add_flags(flags)


class Img2ImgRunner(Txt2ImgRunner):
    def build_command(self):
        super().build_command()

        self.output_path = (
            os.path.join(
                config.get('img2img_dir'),
                f"{self._get_param('in_output')}.png"
            )
            if self._get_param('in_output')
            else os.path.join(
                config.get('img2img_dir'), get_next_img(subctrl=1)
            )
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
            '--style-ratio': (self._get_param('in_style_ratio')
                              if self._get_param('in_style_ratio_bool')
                              else None),
        }
        self._add_options(options)


class Any2VideoRunner(CommandRunner):
    def _add_base_args(self):
        # Override to add video-specific base arguments
        super()._add_base_args()
        self.command.extend([
            '--video-frames', str(self._get_param('in_frames')),
            '--fps', str(self._get_param('in_fps')),
        ])

    def build_command(self):
        self._resolve_paths()
        self.output_path = (
            os.path.join(
                config.get('any2video_dir'),
                f"{self._get_param('in_output')}.avi"
            )
            if self._get_param('in_output')
            else os.path.join(
                config.get('any2video_dir'), get_next_img(subctrl=2)
            )
        )

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
    def build_command(self):
        self._resolve_paths()
        self.output_path = (
            os.path.join(
                config.get('upscale_dir'),
                f"{self._get_param('in_output')}.png"
            )
            if self._get_param('in_output')
            else os.path.join(
                config.get('upscale_dir'), get_next_img(subctrl=3)
            )
        )

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
            '--color': self._get_param('in_color'),
            '-v': self._get_param('in_verbose')
        }
        self._add_flags(flags)


def txt2img(params: dict) -> Generator:
    """Creates and runs a Txt2ImgRunner from a params dictionary."""
    runner = Txt2ImgRunner(mode="img_gen", params=params)
    runner.build_command()
    yield from runner.run()


def img2img(params: dict) -> Generator:
    """Creates and runs an Img2ImgRunner."""
    runner = Img2ImgRunner(mode="img_gen", params=params)
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
    in_orig_model: str, in_model_dir: str, in_quant_type: str,
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
    if in_verbose:
        command.append('-v')

    fcommand = ' '.join(command)
    print(f"\n\n{fcommand}\n\n")

    # This assumes run_subprocess can handle synchronous execution
    # and will block until the process is complete.
    for _ in subprocess_manager.run_subprocess(command):
        pass  # Consume generator if it's async

    return "Process completed."

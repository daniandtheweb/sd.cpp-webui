"""sd.cpp-webui - core - stable-diffusion.cpp cli"""

import os
import re
import sys
import subprocess
from typing import Dict, Any, Generator

import gradio as gr

from modules.core.common.sd_common import (
    DiffusionMode, CommonRunner, process_editor_mask
)
from modules.utils.sdcpp_utils import generate_output_filename
from modules.shared_instance import (
    config, subprocess_manager, SD_CLI
)


class CommandRunner(CommonRunner):
    """Builds and runs stable-diffusion.cpp commands and yields UI updates."""

    def __init__(self, mode: str, params: Dict[str, Any]):
        super().__init__(params)
        self.mode = mode
        self.command = [SD_CLI, '-M', self.mode]
        self.outputs = []
        self.output_path = ""
        self.preview_path = None

    def _make_relative(self, path):
        """Converts absolute paths to be relative to the executable directory."""
        if not path or not os.path.isabs(str(path)):
            return path
        try:
            exe_dir = os.path.dirname(os.path.abspath(SD_CLI))
            return os.path.relpath(str(path), start=exe_dir)
        except ValueError:
            # Fallback for cross-drive paths on Windows
            return path

    def _get_next_synced_index(self) -> int:
        """Finds the next available sequential index by scanning both prompt folders."""
        pp_dir = os.path.join('outputs', 'pprompts')
        np_dir = os.path.join('outputs', 'nprompts')
        os.makedirs(pp_dir, exist_ok=True)
        os.makedirs(np_dir, exist_ok=True)

        max_idx = -1
        for d in [pp_dir, np_dir]:
            for f in os.listdir(d):
                # Updated to recognize both pprompt_ and nprompt_ prefixes
                if (f.startswith('pprompt_') or f.startswith('nprompt_')) and f.endswith('.txt'):
                    try:
                        # Strip prefix and extension to get the number
                        num_str = f.replace('pprompt_', '').replace('nprompt_', '').replace('.txt', '')
                        idx = int(num_str)
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        continue
        return max_idx + 1

    def _save_prompts(self) -> tuple:
        """Saves positive and negative prompts to synced sequential files (skips if empty)."""
        idx = self._get_next_synced_index()
        pp_dir = os.path.join('outputs', 'pprompts')
        np_dir = os.path.join('outputs', 'nprompts')

        pp_text = str(self._get_param('in_pprompt', "")).strip()
        np_text = str(self._get_param('in_nprompt', "")).strip()

        # Only create paths if text exists
        pp_path = os.path.join(pp_dir, f"pprompt_{idx}.txt") if pp_text else None
        np_path = os.path.join(np_dir, f"nprompt_{idx}.txt") if np_text else None

        if pp_path:
            with open(pp_path, 'w', encoding='utf-8') as f:
                f.write(pp_text)
        if np_path:
            with open(np_path, 'w', encoding='utf-8') as f:
                f.write(np_text)

        return pp_path, np_path

    def _set_output_path(self, dir_key: str, subctrl_id: int, extension: str):
        """Determines and sets the output path for the command."""
        output_dir = config.get(dir_key)
        filename_override = self._get_param('in_output')
        output_scheme = config.get('def_output_scheme')

        if filename_override and str(filename_override).strip():
            base_name = str(filename_override).strip()
            filename = f"{base_name}.{extension}"
            test_path = os.path.join(output_dir, filename)

            counter = 1
            while os.path.exists(test_path):
                filename = f"{base_name}_{counter}.{extension}"
                test_path = os.path.join(output_dir, filename)
                counter += 1

            self.output_path = self._make_relative(test_path)
            return

        name_parts = []

        if config.get('def_output_steps'):
            steps_val = self._get_param('in_steps')
            if steps_val:
                name_parts.append(f"{steps_val}_steps")

        if config.get('def_output_quant'):
            quant_val = self._get_param('in_model_type')
            if quant_val and quant_val != "Default":
                name_parts.append(str(quant_val))

        self.output_path = self._make_relative(generate_output_filename(
            output_dir, output_scheme, extension,
            name_parts, subctrl_id
        ))

    def _add_base_args(self):
        """Adds arguments common to all modes."""
        self.command.extend([
            '--sampling-method', str(self._get_param('in_sampling')),
            '--steps', str(self._get_param('in_steps')),
            '-W', str(self._get_param('in_width')),
            '-H', str(self._get_param('in_height')),
            '--cfg-scale', str(self._get_param('in_cfg')),
            '-s', str(self._get_param('in_seed')),
            '-o', self.output_path
            # --output-begin-idx - to implement
        ])

        # Only add -b if batch count differs from default (1)
        batch_count = self._get_param('in_batch_count')
        if batch_count and str(batch_count) != "1":
            self.command.extend(['-b', str(batch_count)])

        # Only add --clip-skip if it differs from default (-1)
        clip_skip = self._get_param('in_clip_skip')
        if clip_skip and str(clip_skip) != "-1":
            self.command.extend(['--clip-skip', str(clip_skip)])

        self.command.extend([
            '--embd-dir', self._make_relative(config.get('emb_dir')),
        ])

        rng = str(self._get_param('in_rng'))
        if rng and str(rng) != "Default":
            self.command.extend([
                '--rng', str(self._get_param('in_rng'))
            ])

        sampler_rng = str(self._get_param('in_sampler_rng'))
        if sampler_rng and str(sampler_rng) != "Default":
            self.command.extend([
                '--sampler-rng', str(self._get_param('in_sampler_rng')),
            ])

        # Only add -t if it differs from default (0)
        threads = self._get_param('in_threads')
        if threads and str(threads) != "0":
            self.command.extend(['-t', str(self._get_param('in_threads'))])

        # Only add LoRA arguments if prompts contain <lora:name:strength> tags
        pp_text = str(self._get_param('in_pprompt', "")).strip()
        np_text = str(self._get_param('in_nprompt', "")).strip()
        if re.search(r'<lora:[^:]+:[^>]+>', f"{pp_text} {np_text}"):
            self.command.extend([
                '--lora-model-dir', self._make_relative(config.get('lora_dir')),
                '--lora-apply-mode', str(self._get_param('in_lora_apply'))
            ])

    def _prepare_for_run(self):
        """
        Prepares the final command string for printing and computes outputs.
        """
        # Prepare a copy for printing (shows actual .txt paths)
        cmd_print = self.command.copy()
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

        process_env = self._build_process_env()

        if self.preview_path:
            gallery_update = [self.preview_path]
        else:
            gallery_update = None

        yield (
            self.fcommand,
            gr.update(visible=True, value=0),
            gr.update(visible=True, value="Initializing..."),
            gr.update(value=""),
            None
        )

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
                if self.preview_path and os.path.isfile(self.preview_path):
                    gallery_update = [self.preview_path]
                else:
                    gallery_update = gr.skip()

                yield (
                    self.fcommand,
                    gr.update(value=update["percent"]),
                    update["status"],
                    gr.update(value=""),
                    gallery_update
                )

        if self.preview_path and os.path.isfile(self.preview_path):
            os.remove(self.preview_path)

        valid_outputs = [out for out in self.outputs if os.path.isfile(out)]

        if valid_outputs:
            final_gallery_update = valid_outputs
        else:
            final_gallery_update = gr.skip()

        yield (
            self.fcommand,
            gr.update(visible=False, value=100),
            gr.update(visible=False, value=""),
            gr.update(value=final_stats_str),
            final_gallery_update
        )


class ImageGenerationRunner(CommandRunner):
    """A common base for txt2img, img2img, and imgedit runners."""

    def _get_model_options(self) -> Dict[str, Any]:
        """Builds and returns a dictionary of model and VAE options."""
        options = {}
        diffusion_mode = self._get_param('in_diffusion_mode')

        if diffusion_mode == DiffusionMode.CHECKPOINT:
            options['--model'] = self._make_relative(self._get_param('f_ckpt_model'))
            options['--vae'] = self._make_relative(self._get_param('f_ckpt_vae'))
        elif diffusion_mode == DiffusionMode.UNET:
            options['--diffusion-model'] = self._make_relative(self._get_param('f_unet_model'))
            options['--vae'] = self._make_relative(self._get_param('f_unet_vae'))
            options['--uncond-diffusion-model'] = self._make_relative(self._get_param('f_uncond_unet_model'))
            options['--clip_g'] = self._make_relative(self._get_param('f_clip_g'))
            options['--clip_l'] = self._make_relative(self._get_param('f_clip_l'))
            options['--t5xxl'] = self._make_relative(self._get_param('f_t5xxl'))
            options['--llm'] = self._make_relative(self._get_param('f_llm'))
            options['--llm_vision'] = self._make_relative(self._get_param('f_llm_vision'))

        # Filter out any keys that have a None value before returning
        return {k: v for k, v in options.items() if v is not None}

    def build_command(self, output_dir_key: str, subctrl_id: int):
        """Builds the common command for image generation."""
        self._resolve_paths()
        self._set_output_path(output_dir_key, subctrl_id, 'png')

        # Save prompts to synced sequential files
        pp_path, np_path = self._save_prompts()
        if pp_path:
            self.command.extend(['--prompt-file', pp_path])
        if np_path:
            self.command.extend(['--negative-prompt-file', np_path])

        self._add_base_args()

        if self._get_param('in_preview_bool'):
            base_name, extension = os.path.splitext(self.output_path)
            self.preview_path = self._make_relative(base_name + "_preview" + extension)

        options = {
            # Models
            **self._get_model_options(),
            # Weight type
            '--type': (self._get_param('in_model_type')
                       if self._get_param('in_model_type') != "Default"
                       else None),
            '--tensor-type-rules': (
                self._get_param('in_tensor_type_rules')
                if self._get_param('in_tensor_type_rules') != ""
                else None
            ),
            # Scheduler
            '--scheduler': (self._get_param('in_scheduler')
                            if not self._get_param('in_sigmas')
                            else None),
            '--sigmas': (self._get_param('in_sigmas')
                         if self._get_param('in_sigmas') != ""
                         else None),
            # TAESD
            '--taesd': self._make_relative(self._get_param('f_taesd')),
            # PhotoMaker
            **({
                '--photo-maker': self._make_relative(self._get_param('f_phtmkr')),
                '--pm-id-images-dir': self._make_relative(self._get_param('in_phtmkr_id')),
                '--pm-id-embed-path': self._make_relative(self._get_param('in_phtmkr_emb')),
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
                '--upscale-model': self._make_relative(self._get_param('f_upscl')),
                '--upscale-repeats': self._get_param('in_upscl_rep'),
                '--upscale-tile-size': self._get_param('in_upscl_tile_size'),

            } if self._get_param('in_upscl_bool') else {}),
            # ControlNet
            **({
                '--control-net': self._make_relative(self._get_param('f_cnnet')),
                '--control-image': self._make_relative(self._get_param('in_control_img')),
                '--control-strength': self._get_param('in_control_strength')
            } if self._get_param('in_cnnet_bool') else {}),
            # Chroma
            '--chroma-t5-mask-pad': (self._get_param('in_t5_mask_pad')
                                     if self._get_param('in_enable_t5_mask')
                                     else None),
            # Skip Layer Guidance (SLG)
            **({
                '--slg-scale': self._get_param('in_slg_scale'),
                '--skip-layer-start': self._get_param('in_skip_layer_start'),
                '--skip-layer-end': self._get_param('in_skip_layer_end'),
                '--skip-layers': (self._get_param('in_skip_layers')
                                  if self._get_param('in_skip_layers') != ""
                                  else None),
            } if self._get_param('in_slg_bool') else {}),
            # Performance
            '--max-vram': (self._get_param('in_max_vram')
                           if self._get_param('in_max_vram') != 0
                           else None),
            # VAE Tiling
            **({
                '--vae-tile-overlap': self._get_param('in_vae_tile_overlap'),
                '--vae-tile-size': (
                    f"{size}x{size}"
                    if (size := self._get_param('in_vae_tile_size'))
                    else None
                ),
                '--vae-relative-tile-size': (
                    f"{size}x{size}"
                    if (
                        self._get_param('in_vae_relative_bool') and
                        (size := self._get_param('in_vae_relative_tile_size'))
                    )
                    else None
                ),
            } if self._get_param('in_vae_tiling') else {}),
            # Cache
            **({
                '--cache-mode': self._get_param('in_cache_mode'),
                '--cache-option': (
                    val.strip('"')
                    if (val := self._get_param('in_cache_option')) and str(val).strip('"') != ""
                    else None
                ),
                '--scm-mask': (self._get_param('in_scm_mask')
                               if self._get_param('in_scm_mask') != ""
                               else None),
                '--scm-policy': (self._get_param('in_scm_policy')
                                 if self._get_param('in_scm_policy') != "none"
                                 else None)
            } if self._get_param('in_cache_bool') else {}),
            # Prediction type override
            '--prediction': (self._get_param('in_predict')
                             if self._get_param('in_predict') != "Default"
                             else None),
            # Preview
            **({
                '--preview': self._get_param('in_preview_mode'),
                '--preview-path': self._make_relative(self.preview_path),
                '--preview-interval': self._get_param('in_preview_interval'),
            } if self._get_param('in_preview_bool') else {})
        }
        self._add_options(options)

        flags = self._get_common_flags()
        flags.update({
            '--taesd-preview-only': (self._get_param('in_preview_taesd')
                                     if self._get_param('in_preview_bool')
                                     else False),
            '--preview-noisy': (self._get_param('in_preview_noisy')
                                if self._get_param('in_preview_bool')
                                else False),
            '--increase-ref-index': self._get_param('in_increase_ref_index'),
            '--disable-auto-resize-ref-image': self._get_param(
                'in_disable_auto_resize_ref_image'
            ),
        })
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
        self.command.extend(['--init-img', self._make_relative(str(self._get_param('in_img_inp')))])
        self.command.extend([
            '--strength',
            str(self._get_param('in_strength'))
        ])

        mask_input = self._get_param('in_img_mask') or self._get_param('in_mask')
        mask_img = process_editor_mask(mask_input)
        final_mask_path = None

        if mask_img is not None:
            # The CLI requires a file path, so we save the PIL Image to disk
            output_dir = config.get('img2img_dir')
            final_mask_path = os.path.join(output_dir, "sdcpp_temp_mask.png")

            try:
                mask_img.save(final_mask_path)
            except Exception as e:
                print(f"Error saving temporary mask for CLI: {e}")
                final_mask_path = None
        options = {
            '--mask': self._make_relative(final_mask_path),
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

        ref_imgs = self._get_param('in_ref_img')

        if not ref_imgs:
            return

        if not isinstance(ref_imgs, list):
            ref_imgs = [ref_imgs]

        for img in ref_imgs:
            if isinstance(img, tuple):
                img_path = img[0]
            elif isinstance(img, dict) and "name" in img:
                img_path = img["name"]
            else:
                img_path = str(img)

            self.command.extend(
                ['--ref-image', self._make_relative(img_path)]
            )


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
        self._set_output_path('any2video_dir', 3, 'webm')

        # Save prompts to synced sequential files
        pp_path, np_path = self._save_prompts()
        self.command.extend(['--prompt-file', pp_path])
        if self._get_param('in_nprompt', "").strip():
            self.command.extend(['--negative-prompt-file', np_path])

        self._add_base_args()

        init_img = (self._get_param('in_img_inp')
                    or self._get_param('in_first_frame_inp'))

        use_high_noise = self._get_param('in_high_noise_bool')
        high_noise_model = self._get_param('f_high_noise_model')

        options = {
            # VAE
            '--vae': self._make_relative(self._get_param('f_unet_vae')),
            '--audio-vae': self._make_relative(self._get_param('f_audio_vae')),
            # Wan2.1, Wan2.2
            '--diffusion-model': self._make_relative(self._get_param('f_unet_model')),
            '--clip_vision': self._make_relative(self._get_param('f_clip_vision_h')),
            '--t5xxl': self._make_relative(self._get_param('f_umt5_xxl')),
            # LTX-2.3
            '--llm': self._make_relative(self._get_param('f_llm')),
            '--embeddings-connectors': self._make_relative(self._get_param('f_emb_connect')),
            # Wan2.2 High Noise Configuration
            '--high-noise-diffusion-model': (
                self._make_relative(high_noise_model)
                if use_high_noise else None
            ),
            '--high-noise-cfg-scale': (
                self._get_param('in_high_noise_cfg')
                if use_high_noise else None
            ),
            '--high-noise-sampling-method': (
                self._get_param('in_high_noise_sampling')
                if use_high_noise else None
            ),
            '--high-noise-steps': (
                self._get_param('in_high_noise_steps')
                if use_high_noise else None
            ),
            '--high-noise-img-cfg-scale': (
                self._get_param('in_high_noise_img_cfg')
                if use_high_noise else None
            ),
            '--high-noise-guidance': (
                self._get_param('in_high_noise_guidance')
                if use_high_noise else None
            ),
            '--high-noise-slg-scale': (
                self._get_param('in_high_noise_slg_scale')
                if use_high_noise else None
            ),
            '--high-noise-skip-layer-start': (
                self._get_param('in_high_noise_skip_layer_start')
                if use_high_noise else None
            ),
            '--high-noise-skip-layer-end': (
                self._get_param('in_high_noise_skip_layer_end')
                if use_high_noise else None
            ),
            '--high-noise-eta': (
                self._get_param('in_high_noise_eta')
                if use_high_noise else None
            ),
            '--high-noise-skip-layers': (
                self._get_param('in_high_noise_skip_layers')
                if use_high_noise else None
            ),
            # Skip Layer Guidance (SLG)
            **({
                '--slg-scale': self._get_param('in_slg_scale'),
                '--skip-layer-start': self._get_param('in_skip_layer_start'),
                '--skip-layer-end': self._get_param('in_skip_layer_end'),
                '--skip-layers': (self._get_param('in_skip_layers')
                                  if self._get_param('in_skip_layers') != ""
                                  else None),
            } if self._get_param('in_slg_bool') else {}),
            # Wan & MoE Specifics
            '--moe-boundary': (self._get_param('in_moe_boundary')
                               if self._get_param('in_moe_boundary_bool')
                               else None),
            '--vace-strength': (self._get_param('in_vace_strength')
                                if self._get_param('in_vace_strength_bool')
                                else None),
            # Inputs for I2V, FLF2V, and VACE V2V
            '--init-img': self._make_relative(init_img),
            '--end-img': self._make_relative(self._get_param('in_last_frame_inp')),
            '--control-video': self._make_relative(self._get_param('in_control_video_dir')),
            # TAESD
            '--taesd': self._get_param('f_taesd'),
            # Upscaling
            '--upscale-model': (self._get_param('f_upscl')
                                if self._get_param('in_upscl_bool')
                                else None),
            '--upscale-repeats': (self._get_param('in_upscl_rep')
                                  if self._get_param('in_upscl_bool')
                                  else None),
            '--upscale-tile-size': (self._get_param('in_upscl_tile_size')
                                    if self._get_param('in_upscl_bool')
                                    else None),
            # Additional Params
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
            '--control-image': (self._make_relative(self._get_param('in_control_img'))
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

        flags = self._get_common_flags()
        flags.update({
            '--taesd-preview-only': (self._get_param('in_preview_taesd')
                                     if self._get_param('in_preview_bool')
                                     else False),
            '--preview-noisy': (self._get_param('in_preview_noisy')
                                if self._get_param('in_preview_bool')
                                else False),
            # LTX Specifics
            '--temporal-tiling': self._get_param('in_temporal_tiling'),
        })
        self._add_flags(flags)


class UpscaleRunner(CommandRunner):
    """Builds the upscale command."""

    def build_command(self):
        self._resolve_paths()
        self._set_output_path('upscale_dir', 4, 'png')

        init_img = (self._get_param('in_img_inp')
                    or self._get_param('in_first_frame_inp'))
        options = {
            '--init-img': self._make_relative(init_img),
            '--upscale-model': self._make_relative(self._get_param('f_upscl')),
            '-W': self._get_param('in_init_width'),
            '-H': self._get_param('in_init_height'),
            '--upscale-repeats': self._get_param('in_upscl_rep'),
            '--upscale-tile-size': self._get_param('in_upscl_tile_size'),
            '-o': self.output_path,
        }
        self._add_options(options)

        self._add_flags(self._get_common_flags())


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


def convert(params: dict):
    """
    Runs the model conversion command.
    Outputs to terminal (raw) and UI log (cleaned of ANSI codes).
    """
    in_orig_model = params.get('in_orig_model')
    in_model_dir = params.get('in_model_dir')
    in_quant_type = params.get('in_quant_type')
    in_tensor_type_rules = params.get('in_tensor_type_rules')
    in_convert_name = params.get('in_convert_name', False)
    in_gguf_name = params.get('in_gguf_name')
    in_color = params.get('in_color', True)
    in_verbose = params.get('in_verbose', False)

    exe_dir = os.path.dirname(os.path.abspath(SD_CLI))

    orig_model_path = os.path.relpath(os.path.join(in_model_dir, in_orig_model), start=exe_dir)

    if in_gguf_name:
        if not in_gguf_name.endswith('.gguf'):
            in_gguf_name += '.gguf'
        gguf_path = os.path.relpath(os.path.join(in_model_dir, in_gguf_name), start=exe_dir)
    else:
        model_name, _ = os.path.splitext(in_orig_model)
        gguf_path = os.path.relpath(
            os.path.join(in_model_dir, f"{model_name}-{in_quant_type}.gguf"), start=exe_dir
        )

    command = [
        SD_CLI, '-M', 'convert',
        '--model', orig_model_path,
        '-o', gguf_path,
        '--type', in_quant_type
    ]

    if in_tensor_type_rules:
        command.extend(['--tensor-type-rules', in_tensor_type_rules])
    if in_convert_name:
        command.append('--convert-name')
    if in_color:
        command.append('--color')
    if in_verbose:
        command.append('-v')

    fcommand = ' '.join(command)
    print(f"\n\n{fcommand}\n\n")

    progress_regex = re.compile(r'(\d+)/(\d+)')

    yield (
        fcommand,
        gr.update(visible=True, value=0),
        gr.update(visible=True, value="Initializing conversion..."),
    )

    last_was_progress = False

    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        ) as process:

            subprocess_manager.process = process

            for output_line in process.stdout:
                output_line = output_line.rstrip()

                match = progress_regex.search(output_line)

                if match:
                    current, total = map(int, match.groups())
                    percent = (current / total) * 100 if total > 0 else 0

                    yield (
                        fcommand,
                        gr.Slider(value=percent),
                        gr.Textbox(value="Converting..."),
                    )

                    sys.stdout.write(f"\r{output_line}")
                    sys.stdout.flush()
                    last_was_progress = True
                else:
                    if last_was_progress:
                        print("\n")
                        last_was_progress = False
                    print(output_line)

    except Exception as e:
        print(f"\nError: {e}")
        yield (
            fcommand,
            gr.Slider(visible=False),
            gr.Textbox(value=f"Error: {e}"),
        )

    finally:
        # cleanup
        if last_was_progress:
            print("\n")

        if subprocess_manager.process:
            subprocess_manager.process = None

    yield (
        fcommand,
        gr.Slider(visible=False),
        gr.Textbox(value="Done."),
    )

"""sd.cpp-webui - core - stable-diffusion.cpp cli"""

import os
import re
import sys
import subprocess
from typing import Dict, Any, Generator

import gradio as gr

from modules.core.common.sd_common import (
    CommonRunner, process_editor_mask,
    has_output_index_spec, expand_output_index
)
from modules.options import emit
from modules.utils.math_utils import remember_seed
from modules.ui.progress import progress_update
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
        # Prompt files (positive, negative), see _save_prompts
        self._prompt_paths = (None, None)
        self.run_idx = self._get_next_synced_index()

    def _get_next_synced_index(self) -> int:
        """Finds the next available sequential index by scanning prompt and command folders."""
        dirs = [
            os.path.join('outputs', 'pprompts'),
            os.path.join('outputs', 'nprompts'),
            os.path.join('outputs', 'commands')
        ]

        max_idx = -1
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                if (f.startswith('pprompt_') or f.startswith('nprompt_') or f.startswith('command_')) and f.endswith('.txt'):
                    try:
                        # Strip all possible prefixes and the extension to get the number
                        num_str = f.replace('pprompt_', '').replace('nprompt_', '').replace('command_', '').replace('.txt', '')
                        idx = int(num_str)
                        if idx > max_idx:
                            max_idx = idx
                    except ValueError:
                        continue
        return max_idx + 1

    def _save_prompts(self) -> tuple:
        """Saves positive and negative prompts to synced sequential files (skips if empty)."""
        idx = self.run_idx
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

    def _save_command(self):
        """Saves the final generated CLI command to a synced file."""
        cmd_dir = os.path.join('outputs', 'commands')
        cmd_path = os.path.join(cmd_dir, f"command_{self.run_idx}.txt")

        with open(cmd_path, 'w', encoding='utf-8') as f:
            f.write(self.fcommand)

    def _indexed_output(self) -> bool:
        """
        True when the outputs are numbered: a batch, or a %d index
        specifier in the output path (--output-begin-idx then applies).
        """
        batch_count = self._get_param('in_batch_count')
        multi_output = (batch_count is not None
                        and int(float(batch_count)) > 1)
        return multi_output or has_output_index_spec(self.output_path)

    def _run_values(self) -> Dict[str, Any]:
        """The values the registry can't know (see Option.source 'ctx:')."""
        return {'output_path': self.output_path}

    def _add_registry_args(self):
        """Adds every registered argument (modules/options/registry.py)."""
        context = self._context(
            values=self._run_values(),
            flags={'indexed_output': self._indexed_output()}
        )
        self.command.extend(emit.cli(context))

    def _prepare_for_run(self):
        """
        Prepares the final command string for printing and computes outputs.
        """
        # Prepare a copy for printing (shows actual .txt paths)
        cmd_print = self.command.copy()
        self.fcommand = ' '.join(map(str, cmd_print))

        # Compute all output filenames. The Gradio slider yields floats, so
        # normalize to int. Mirrors sd-cli (stable-diffusion.cpp
        # examples/cli/main.cpp): with a %d/%Nd specifier in the path, each
        # output is the path expanded with the batch index (offset by the
        # output begin index). Without one, batch > 1 writes suffixed files
        # (<base>_<begin+i>.png, 0-based) — no unsuffixed file.
        batch_count = int(float(self._get_param('in_batch_count', 1) or 1))
        # Qwen Image Layered decodes layers + 1 images per generation,
        # numbered like a batch
        layers = self._qwen_image_layers()
        if layers is not None:
            batch_count *= layers + 1
        # sd-cli clamps negative begin indices to 0 (main.cpp:482)
        begin_idx = max(0, self._int_param('in_output_begin_idx') or 0)
        if has_output_index_spec(self.output_path):
            self.outputs = [
                expand_output_index(self.output_path, begin_idx + i)
                for i in range(batch_count)
            ]
        elif batch_count <= 1:
            self.outputs = [self.output_path]
        else:
            base, ext = os.path.splitext(self.output_path)
            self.outputs = [
                f"{base}_{begin_idx + i}{ext}" for i in range(batch_count)
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
        self._save_command()
        print(f"\n\n{self.fcommand}\n\n")
        remember_seed(emit.cli_value('seed', self.command))

        process_env = self._build_process_env()

        if self.preview_path:
            gallery_update = [self.preview_path]
        else:
            gallery_update = None

        yield (
            self.fcommand,
            progress_update(0, status="Initializing..."),
            gr.update(visible=False, value=""),
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
                    progress_update(
                        update["percent"],
                        update.get("step"), update.get("total"),
                        update["status"]
                    ),
                    gr.skip(),
                    gallery_update
                )

        if self.preview_path and os.path.isfile(self.preview_path):
            os.remove(self.preview_path)

        # Stopped by the user: keep the command, clear the rest
        if subprocess_manager.stopped:
            yield (
                self.fcommand,
                progress_update(visible=False),
                gr.update(visible=False, value=""),
                gr.update(value=None)
            )
            return

        valid_outputs = [out for out in self.outputs if os.path.isfile(out)]

        if valid_outputs:
            final_gallery_update = valid_outputs
        else:
            final_gallery_update = gr.skip()

        yield (
            self.fcommand,
            progress_update(100, visible=False),
            # Statistics are only shown at the end
            gr.update(visible=True, value=final_stats_str),
            final_gallery_update
        )


class ImageGenerationRunner(CommandRunner):
    """A common base for txt2img, img2img, and imgedit runners."""

    def _run_values(self) -> Dict[str, Any]:
        values = super()._run_values()
        values.update(
            prompt_file=self._prompt_paths[0],
            negative_prompt_file=self._prompt_paths[1],
            preview_path=self.preview_path,
        )
        return values

    def build_command(self, output_dir_key: str, subctrl_id: int):
        """Builds the common command for image generation."""
        self._resolve_paths()
        self._set_output_path(output_dir_key, subctrl_id, 'png')

        # Save prompts to synced sequential files
        self._prompt_paths = self._save_prompts()

        if self._get_param('in_preview_bool'):
            base_name, extension = os.path.splitext(self.output_path)
            self.preview_path = self._make_relative(base_name + "_preview" + extension)

        self._add_registry_args()


class Txt2ImgRunner(ImageGenerationRunner):
    """Builds the txt2img command."""

    kind = 'txt2img'

    def build_command(self):
        super().build_command(
            output_dir_key='txt2img_dir', subctrl_id=0
        )


class Img2ImgRunner(ImageGenerationRunner):
    """Builds the img2img command."""

    kind = 'img2img'

    def _save_mask(self) -> str | None:
        """
        The CLI requires a file path, so the mask image is saved to disk.
        Returns its path, None without a mask.
        """
        mask_input = self._get_param('in_img_mask') or self._get_param('in_mask')
        mask_img = process_editor_mask(mask_input)
        if mask_img is None:
            return None
        output_dir = config.get_or_create_dir('img2img_dir')
        mask_path = os.path.join(output_dir, "sdcpp_temp_mask.png")
        try:
            mask_img.save(mask_path)
        except Exception as e:
            print(f"Error saving temporary mask for CLI: {e}")
            return None
        return mask_path

    def _run_values(self) -> Dict[str, Any]:
        values = super()._run_values()
        values['mask_path'] = self._save_mask()
        return values

    def build_command(self):
        super().build_command(
            output_dir_key='img2img_dir', subctrl_id=1
        )


class ImgEditRunner(ImageGenerationRunner):
    """Builds the image editing (instruct) command."""

    kind = 'imgedit'

    def build_command(self):
        super().build_command(
            output_dir_key='imgedit_dir', subctrl_id=2
        )


class Any2VideoRunner(CommandRunner):
    """Builds the any2video command."""

    kind = 'video'

    def _run_values(self) -> Dict[str, Any]:
        values = super()._run_values()
        values.update(
            prompt_file=self._prompt_paths[0],
            negative_prompt_file=self._prompt_paths[1],
        )
        return values

    def build_command(self):
        self._resolve_paths()
        self._set_output_path('any2video_dir', 3, 'webm')

        # Save prompts to synced sequential files
        self._prompt_paths = self._save_prompts()

        self._add_registry_args()


class UpscaleRunner(CommandRunner):
    """Builds the upscale command."""

    kind = 'upscale'

    def build_command(self):
        self._resolve_paths()
        self._set_output_path('upscale_dir', 4, 'png')

        self._add_registry_args()


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
    in_gguf_name = params.get('in_gguf_name')

    # Validate required inputs up front: a missing model dir in the
    # config (ConfigManager returns None) or an empty model selection
    # would otherwise crash os.path.join, and a missing quant type
    # would emit `--type None`.
    if not in_model_dir or not in_orig_model or not in_quant_type:
        error = ("Conversion failed: model directory, model and "
                 "quantization type are all required. Make sure the "
                 "model folder exists in your config and a model is "
                 "selected.")
        print(f"\nError: {error}\n")
        yield (
            "",
            progress_update(visible=False),
            gr.update(visible=True, value=f"Error: {error}"),
        )
        return

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

    command = [SD_CLI, '-M', 'convert', *emit.cli(emit.Context(
        params, 'convert',
        values={'model_path': orig_model_path, 'output_path': gguf_path}
    ))]

    fcommand = ' '.join(command)
    print(f"\n\n{fcommand}\n\n")

    # Only sd.cpp's real convert progress lines look like
    # `loading 123/456 tensors from ...`; a bare N/M match would
    # swallow unrelated log lines (resolutions, step counters, ...).
    progress_regex = re.compile(r'\b(\d+)/(\d+)\s+tensors\b')

    yield (
        fcommand,
        progress_update(
            0, status="Initializing conversion...", unit="Tensor"
        ),
        gr.update(visible=False, value=""),
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

                if match and int(match.group(1)) <= int(match.group(2)):
                    current, total = map(int, match.groups())
                    percent = (current / total) * 100 if total > 0 else 0

                    yield (
                        fcommand,
                        progress_update(
                            percent, current, total,
                            status="Converting...", unit="Tensor"
                        ),
                        gr.skip(),
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
            progress_update(visible=False),
            gr.update(visible=True, value=f"Error: {e}"),
        )

    finally:
        # cleanup
        if last_was_progress:
            print("\n")

        if subprocess_manager.process:
            subprocess_manager.process = None

    yield (
        fcommand,
        progress_update(visible=False),
        gr.update(visible=True, value="Done."),
    )

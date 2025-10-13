"""sd.cpp-webui - Gallery module"""

import os
import re
import json
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr

from modules.shared_instance import config


class GalleryManager:
    """Controls the gallery block"""

    def __init__(
        self, txt2img_gallery: str, img2img_gallery: str,
        any2video_gallery: str, upscale_gallery: str
    ):
        self.dirs: List[str] = [
            txt2img_gallery, img2img_gallery, any2video_gallery,
            upscale_gallery
        ]
        self.page_num: int = 1
        self.ctrl: int = 0
        self.selected_img_index_on_page: Optional[int] = None
        self.selected_img_global_index: Optional[int] = None
        self.current_img_path: Optional[str] = None

    def _get_current_dir(self) -> str:
        """Determines the directory based on the control value."""
        if 0 <= self.ctrl < len(self.dirs):
            return self.dirs[self.ctrl]
        return self.dirs[0]

    def _get_sorted_files(self) -> List[str]:
        """
        Gets all image files from the current directory,
        sorted by creation time.
        """
        img_dir = self._get_current_dir()
        try:
            files = (
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.avi'))
            )
            return sorted(files, key=os.path.getctime)
        except (FileNotFoundError, OSError):
            return []

    def _parse_png_metadata(self, img_path: str) -> Optional[str]:
        """Reads the tEXt chunk from a PNG file to get metadata."""
        try:
            with open(img_path, 'rb') as file:
                if file.read(8) != b'\x89PNG\r\n\x1a\n':
                    return None  # Not a valid PNG

                while True:
                    length_chunk = file.read(4)
                    if not length_chunk:
                        break
                    length = int.from_bytes(length_chunk, byteorder='big')
                    chunk_type = file.read(4).decode('utf-8', errors='ignore')

                    if chunk_type == 'tEXt':
                        # Found the metadata chunk
                        png_block = file.read(length)
                        _ = file.read(4)  # Skip CRC
                        keyword, value = png_block.split(b'\x00', 1)
                        return (
                            f"PNG: tEXt\n{keyword.decode('utf-8')}: "
                            f"{value.decode('utf-8')}"
                        )

                    file.seek(length + 4, 1)  # Skip chunk data and CRC

        except Exception:
            return None
        return None

    def _parse_jpg_metadata(self, img_path: str) -> Optional[str]:
        """Extracts UserComment EXIF data from a JPG file."""
        try:
            with Image.open(img_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    exif = exif_data.get(37510)  # 37510 = UserComment tag
                    if exif:
                        return (
                            f"JPG: Exif\nPositive prompt: "
                            f"{exif.decode('utf-8', errors='ignore')[8:]}"
                        )
                    return "JPG: No User Comment found."
        except Exception:
            return "JPG: No EXIF data found."
        return "JPG: No EXIF data found."

    def _parse_comfyui_workflow(
            self, text_data: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parses a ComfyUI JSON workflow using match-case
        and a data-driven approach.
        """
        try:
            json_start_index = text_data.find('{')
            if json_start_index == -1:
                return None

            workflow_data = json.loads(text_data[json_start_index:])
            params = {}
            found_positive = False

            keys_to_extract = ["steps", "cfg", "sampler_name", "scheduler"]

            for node in workflow_data.values():
                if not isinstance(node, dict):
                    continue

                class_type = node.get("class_type")
                inputs = node.get("inputs", {})

                match class_type:
                    case "CLIPTextEncode":
                        meta_title = (
                            node.get("_meta", {}).get("title", "").lower()
                        )
                        if "positive" in meta_title and not found_positive:
                            params['pprompt'] = inputs.get("text")
                            found_positive = True
                        elif "negative" in meta_title:
                            params['nprompt'] = inputs.get("text")

                for key in keys_to_extract:
                    if key in inputs:
                        params[key] = inputs[key]

                if "noise_seed" in inputs:
                    params['seed'] = inputs["noise_seed"]
                elif "seed" in inputs:
                    params['seed'] = inputs["seed"]

                if "sampler_name" in inputs:
                    params['sampler'] = inputs['sampler_name']

            return params if any(params.values()) else None

        except (json.JSONDecodeError, AttributeError):
            return None

    def _parse_a1111_text(self, text_data: str) -> Dict[str, Any]:
        """
        Parses A1111-style text metadata, now separating sampler and scheduler
        based on the first space.
        """
        params = {}

        patterns = {
            'pprompt': (
                r'(?s)(?:Positive prompt|parameters):\s*(.*?)'
                r'(?=\s*(?:Negative prompt:|Steps:|CFG scale:|Seed:|'
                r'Size:|Model:|Sampler:|$))'
            ),
            'nprompt': (
                r'(?s)Negative prompt:\s*(.*?)'
                r'(?=\s*(?:Steps:|CFG scale:|Seed:|Size:|Model:|Sampler:|$))'
            ),
            'steps': r'Steps:\s*(\d+)',
            'cfg': r'CFG scale:\s*([\d.]+)',
            'seed': r'Seed:\s*(\d+)',
        }
        converters = {'steps': int, 'cfg': float, 'seed': int}

        for key, pattern in patterns.items():
            match = re.search(
                pattern,
                text_data,
                re.IGNORECASE
            )
            if match:
                value = match.group(1).strip()
                if key not in ['pprompt', 'nprompt']:
                    value = value.split(',')[0]

                params[key] = converters.get(key, lambda x: x)(value)

        sampler_match = re.search(
            r'Sampler:\s*([^,]+)', text_data, re.IGNORECASE
        )
        if sampler_match:
            full_sampler_str = sampler_match.group(1).strip()
            parts = full_sampler_str.split(' ', 1)
            params['sampler'] = parts[0]
            if len(parts) > 1:
                params['scheduler'] = parts[1]
            else:
                params['scheduler'] = ""

        return params

    def _extract_params_from_text(self, text_data: str) -> Dict[str, Any]:
        """
        Extracts generation parameters by dispatching to the correct parser.
        This function now has a cyclomatic complexity of just ~3.
        """
        default_params = {
            'pprompt': "", 'nprompt': "", 'steps': None, 'sampler': "",
            'scheduler': "", 'cfg': None, 'seed': None
        }

        if not text_data:
            return default_params

        comfy_params = self._parse_comfyui_workflow(text_data)
        if comfy_params:
            return {**default_params, **comfy_params}

        a1111_params = self._parse_a1111_text(text_data)
        return {**default_params, **a1111_params}

    def reload_gallery(
        self, page_num: int = 1, ctrl_inp: Optional[int] = None
    ) -> Tuple[List[Image.Image], int, gr.Gallery]:
        """Reloads the gallery block to a specific page."""
        files = self._get_sorted_files()
        total_imgs = len(files)
        total_pages = (total_imgs + 15) // 16 or 1

        try:
             page_num = int(page_num)
        except (ValueError, TypeError):
             page_num = 1

        if page_num > total_pages:
            page_num = total_pages
        elif page_num < 1:
            page_num = 1

        if ctrl_inp is not None:
            self.ctrl = int(ctrl_inp)

        self.page_num = int(page_num)
        files = self._get_sorted_files()

        start_index = (self.page_num - 1) * 16
        end_index = start_index + 16

        page_files = files[start_index:end_index]

        imgs = [Image.open(path) for path in page_files]

        dir_map = {
            0: 'txt2img',
            1: 'img2img',
            2: 'any2video',
            3: 'upscale'
        }
        current_label = dir_map.get(self.ctrl, 'Gallery')

        # Reset selections when reloading
        self.selected_img_index_on_page = None
        self.current_img_path = None

        return imgs, self.page_num, gr.Gallery(selected_index=None), gr.update(label=current_label)

    def _navigate_page(
        self, direction: int
    ) -> Tuple[List[Image.Image], int, gr.Gallery]:
        """Helper for next/prev/last page navigation."""
        files = self._get_sorted_files()
        total_imgs = len(files)
        total_pages = (total_imgs + 15) // 16 or 1

        if direction == 1:  # Next
            self.page_num = (
                self.page_num + 1
                if self.page_num < total_pages
                else 1
            )
        elif direction == -1:  # Previous
            self.page_num = (
                self.page_num - 1
                if self.page_num > 1
                else total_pages
            )
        elif direction == 0:  # Go to last page
            self.page_num = total_pages

        return self.reload_gallery(page_num=self.page_num)

    def next_page(self):
        """Moves to the next gallery page."""
        return self._navigate_page(1)

    def prev_page(self):
        """Moves to the previous gallery page."""
        return self._navigate_page(-1)

    def last_page(self):
        """Moves to the last gallery page."""
        return self._navigate_page(0)

    def get_img_info(self, sel_data: gr.SelectData) -> Tuple[Any, ...]:
        """Reads and parses generation data from a selected image."""
        if not sel_data:
            # Return empty values if no image is selected
            return ("", "", None, None, None, "", "", None, None, "", "")

        self.selected_img_index_on_page = sel_data.index
        # Calculate the global index across all pages
        self.selected_img_global_index = (
            ((self.page_num - 1) * 16) + sel_data.index
        )

        files = self._get_sorted_files()

        if self.selected_img_global_index >= len(files):
            return (
                "", "Image index out of range.", None, None, None, "", "",
                None, None, "", ""
            )

        self.current_img_path = files[self.selected_img_global_index]

        raw_text = None
        file_path_lower = self.current_img_path.lower()

        if file_path_lower.endswith('.png'):
            raw_text = self._parse_png_metadata(self.current_img_path)
        elif file_path_lower.endswith(('.jpg', '.jpeg')):
            raw_text = self._parse_jpg_metadata(self.current_img_path)

        params = self._extract_params_from_text(raw_text)

        try:
            with Image.open(self.current_img_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None

        return (
            params['pprompt'], params['nprompt'], width, height,
            params['steps'], params['sampler'], params['scheduler'],
            params['cfg'], params['seed'], self.current_img_path,
            raw_text or ""
        )

    def delete_img(self) -> Tuple[Any, ...]:
        """
        Deletes the currently selected image. It then selects the previous
        image, or the next if the first was deleted. If the gallery becomes
        empty, it resets the selection.
        """
        # If no image is selected or the index is unknown, just refresh.
        if (not self.current_img_path or
                not os.path.exists(self.current_img_path) or
                self.selected_img_global_index is None):
            imgs, page_num, gallery_update = (
                self.reload_gallery(page_num=self.page_num)
            )
            return (
                imgs, page_num, gallery_update, "", "", None, None, None, "",
                "", None, None, "", ""
            )

        index_to_delete = self.selected_img_global_index
        path_to_delete = self.current_img_path

        try:
            os.remove(path_to_delete)
            print(f"Deleted {path_to_delete}")
        except OSError as e:
            print(f"Error deleting file: {e}")

        files_after_delete = self._get_sorted_files()

        if not files_after_delete:
            self.current_img_path = None
            self.selected_img_index_on_page = None
            self.selected_img_global_index = None

            imgs, page_num, gallery_update = (
                self.reload_gallery(page_num=1)
            )

            return (
                imgs, page_num, gallery_update,
                "", "", None, None, None, "", "", None, None, "", ""
            )

        new_global_index = index_to_delete - 1 if index_to_delete > 0 else 0

        if new_global_index >= len(files_after_delete):
            new_global_index = len(files_after_delete) - 1

        new_page_num = (new_global_index // 16) + 1
        new_page_index = new_global_index % 16

        imgs, page_num, _, _ = self.reload_gallery(page_num=new_page_num)

        gallery_update = gr.Gallery(selected_index=new_page_index)

        self.selected_img_global_index = new_global_index
        self.selected_img_index_on_page = new_page_index
        self.current_img_path = files_after_delete[new_global_index]

        raw_text = None
        if self.current_img_path.lower().endswith('.png'):
            raw_text = self._parse_png_metadata(self.current_img_path)
        elif self.current_img_path.lower().endswith(('.jpg', '.jpeg')):
            raw_text = self._parse_jpg_metadata(self.current_img_path)

        params = self._extract_params_from_text(raw_text)

        try:
            with Image.open(self.current_img_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None

        return (
            imgs, page_num, gallery_update,
            params['pprompt'], params['nprompt'], width, height,
            params['steps'], params['sampler'], params['scheduler'],
            params['cfg'], params['seed'], self.current_img_path,
            raw_text or ""
        )


def get_next_img(subctrl: int) -> str:
    """Creates a new, sequential image name (e.g., '123.png')."""
    dir_map = {
        0: 'txt2img_dir',
        1: 'img2img_dir',
        2: 'any2video_dir',
        3: 'upscale_dir'
    }
    dir_key = dir_map.get(subctrl, 'txt2img_dir')
    img_out_dir = config.get(dir_key)

    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir, exist_ok=True)

    try:
        numbers = [
            int(f[:-4])
            for f in os.listdir(img_out_dir)
            if f.endswith('.png') and f[:-4].isdigit()
        ]
        next_number = max(numbers) + 1 if numbers else 1
    except (ValueError, FileNotFoundError):
        next_number = 1

    return f"{next_number}.png"

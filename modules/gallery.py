"""sd.cpp-webui - Gallery module"""

import os
import re
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr

from modules.shared_instance import config


class GalleryManager:
    """Controls the gallery block"""

    def __init__(
        self, txt2img_gallery: str, img2img_gallery: str,
        any2video_gallery: str
    ):
        self.dirs: List[str] = [
            txt2img_gallery, img2img_gallery, any2video_gallery
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
                        # Decoding can be tricky, this is a common approach
                        return (
                            f"JPG: Exif\nPositive prompt: "
                            f"{exif.decode('utf-8', errors='ignore')[8:]}"
                        )
                    return "JPG: No User Comment found."
        except Exception:
            return "JPG: No EXIF data found."
        return "JPG: No EXIF data found."

    def _extract_params_from_text(self, text_data: str) -> Dict[str, Any]:
        """
        Uses regex to extract generation parameters from a raw metadata string.
        """
        params = {
            'pprompt': "", 'nprompt': "", 'steps': None, 'sampler': "",
            'cfg': None, 'seed': None
        }
        if not text_data:
            return params

        # A1111 / General Format
        pprompt_match = re.search(
            r'(?:Positive prompt|parameters):\s*(.*?)'
            r'(?:\nNegative prompt:|\nSteps:|$)',
            text_data, re.DOTALL
        )
        nprompt_match = re.search(
            r'Negative prompt:\s*(.*?)(?:\nSteps:|$)', text_data, re.DOTALL
        )

        if pprompt_match:
            params['pprompt'] = pprompt_match.group(1).strip()
        if nprompt_match:
            params['nprompt'] = nprompt_match.group(1).strip()

        # Extract other key-value pairs
        steps_match = re.search(r'Steps:\s*(\d+)', text_data)
        sampler_match = re.search(r'Sampler:\s*([\w\s+]+)', text_data)
        cfg_match = re.search(r'CFG scale:\s*([\d.]+)', text_data)
        seed_match = re.search(r'Seed:\s*(\d+)', text_data)

        if steps_match:
            params['steps'] = int(steps_match.group(1))
        if sampler_match:
            params['sampler'] = sampler_match.group(1).strip()
        if cfg_match:
            params['cfg'] = float(cfg_match.group(1))
        if seed_match:
            params['seed'] = int(seed_match.group(1))

        # Fallback for ComfyUI JSON-like format
        if not params['pprompt'] and '"positive_prompt":' in text_data:
            pprompt_json_match = re.search(
                r'"positive_prompt":\s*"([^"]*)"', text_data
            )
            nprompt_json_match = re.search(
                r'"negative_prompt":\s*"([^"]*)"', text_data
            )
            if pprompt_json_match:
                params['pprompt'] = pprompt_json_match.group(1)
            if nprompt_json_match:
                params['nprompt'] = nprompt_json_match.group(1)

        return params

    def reload_gallery(
        self, ctrl_inp: Optional[int] = None, page_num: int = 1
    ) -> Tuple[List[Image.Image], int, gr.Gallery]:
        """Reloads the gallery block to a specific page."""
        if ctrl_inp is not None:
            self.ctrl = int(ctrl_inp)

        self.page_num = int(page_num)
        files = self._get_sorted_files()

        start_index = (self.page_num - 1) * 16
        end_index = start_index + 16

        page_files = files[start_index:end_index]

        imgs = [Image.open(path) for path in page_files]

        # Reset selections when reloading
        self.selected_img_index_on_page = None
        self.current_img_path = None

        return imgs, self.page_num, gr.Gallery(selected_index=None)

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
            return ("", "", None, None, None, "", None, None, "", "")

        self.selected_img_index_on_page = sel_data.index
        # Calculate the global index across all pages
        self.selected_img_global_index = (
            ((self.page_num - 1) * 16) + sel_data.index
        )

        files = self._get_sorted_files()

        if self.selected_img_global_index >= len(files):
            return (
                "", "Image index out of range.", None, None, None, "", None,
                None, "", ""
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
            params['steps'], params['sampler'], params['cfg'], params['seed'],
            self.current_img_path, raw_text or ""
        )

    def delete_img(self) -> Tuple[Any, ...]:
        """Deletes the currently selected image and refreshes the gallery."""
        if (not self.current_img_path or
                not os.path.exists(self.current_img_path)):
            # No image is selected or it's already gone, just refresh
            imgs, page_num, gallery_update = (
                self.reload_gallery(page_num=self.page_num)
            )
            return (
                imgs, page_num, gallery_update, "", "", None, None, None, "",
                None, None, "", ""
            )

        try:
            os.remove(self.current_img_path)
            print(f"Deleted {self.current_img_path}")
        except OSError as e:
            print(f"Error deleting file: {e}")
            # Fall through to refresh the UI anyway

        # The image is gone, so reload everything and reset state
        self.current_img_path = None
        self.selected_img_index_on_page = None

        # Check if the current page is now empty and go to the previous one
        files = self._get_sorted_files()
        total_pages = (len(files) + 15) // 16 or 1
        if self.page_num > total_pages:
            self.page_num = total_pages

        imgs, page_num, gallery_update = (
            self.reload_gallery(page_num=self.page_num)
        )

        # Return a full tuple to clear all outputs
        return (
            imgs, page_num, gallery_update,
            "", "", None, None, None, "", None, None, "", ""
        )


def get_next_img(subctrl: int) -> str:
    """Creates a new, sequential image name (e.g., '123.png')."""
    dir_map = {
        0: 'txt2img_dir',
        1: 'img2img_dir',
        2: 'any2video_dir'
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

"""sd.cpp-webui - Gallery module"""

import os
from typing import List, Tuple, Any, Optional
from PIL import Image

import gradio as gr

from modules.shared_instance import config
from modules.utils.image_utils import size_extractor
from modules.utils.metadata_utils import (
    parse_png_metadata, parse_jpg_metadata,
    extract_params_from_text
)


class GalleryManager:
    """Controls the gallery block"""

    def __init__(
        self, txt2img_gallery: str, img2img_gallery: str,
        imgedit_gallery: str, any2video_gallery: str,
        upscale_gallery: str
    ):
        self.dirs: List[str] = [
            txt2img_gallery, img2img_gallery, imgedit_gallery,
            any2video_gallery, upscale_gallery
        ]
        self.page_num: int = 1
        self.ctrl: int = 0

        self.sort_order: str = config.get('def_gallery_sorting')

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
            if self.sort_order == "Date (Newest First)":
                return sorted(files, key=os.path.getctime, reverse=True)
            elif self.sort_order == "Name (A-Z)":
                return sorted(files, key=lambda x: os.path.basename(x).lower())
            elif self.sort_order == "Name (Z-A)":
                return sorted(files, key=lambda x: os.path.basename(x).lower(), reverse=True)
            else:
                # Default: Date (Oldest First)
                return sorted(files, key=os.path.getctime)

        except (FileNotFoundError, OSError):
            return []

    def reload_gallery(
        self, page_num: int = 1, ctrl_inp: Optional[int] = None,
        sort_inp: Optional[str] = None
    ) -> Tuple[List[Image.Image], int, gr.Gallery]:
        """Reloads the gallery block to a specific page."""

        if sort_inp is not None:
            self.sort_order = sort_inp

        if ctrl_inp is not None:
            self.ctrl = int(ctrl_inp)

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

        self.page_num = int(page_num)

        start_index = (self.page_num - 1) * 16
        end_index = start_index + 16

        page_files = files[start_index:end_index]

        imgs = [Image.open(path) for path in page_files]

        dir_map = {
            0: 'txt2img',
            1: 'img2img',
            2: 'imgedit',
            3: 'any2video',
            4: 'upscale'
        }
        current_label = f"{dir_map.get(self.ctrl, 'Gallery')} - {self.sort_order}"

        # Reset selections when reloading
        self.selected_img_index_on_page = None
        self.current_img_path = None

        return (
            imgs, self.page_num, gr.Gallery(selected_index=None),
            gr.update(label=current_label)
        )

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
            raw_text = parse_png_metadata(self.current_img_path)
        elif file_path_lower.endswith(('.jpg', '.jpeg')):
            raw_text = parse_jpg_metadata(self.current_img_path)

        params = extract_params_from_text(raw_text)

        width, height = size_extractor(self.current_img_path)

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
            imgs, page_num, gallery_update, _ = (
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

            imgs, page_num, gallery_update, _ = (
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
            raw_text = parse_png_metadata(self.current_img_path)
        elif self.current_img_path.lower().endswith(('.jpg', '.jpeg')):
            raw_text = parse_jpg_metadata(self.current_img_path)

        params = extract_params_from_text(raw_text)

        width, height = size_extractor(self.current_img_path)

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
        2: 'imgedit_dir',
        3: 'any2video_dir',
        4: 'upscale_dir'
    }
    dir_key = dir_map.get(subctrl, 'txt2img_dir')
    img_out_dir = config.get(dir_key)

    if not os.path.isdir(img_out_dir):
        os.makedirs(img_out_dir, exist_ok=True)

    try:
        numbers = []
        for f in os.listdir(img_out_dir):
            if f.endswith('.png'):
                name_without_ext = f[:-4]

                base_num_str = name_without_ext.split('_')[0]

                if base_num_str.isdigit():
                    numbers.append(int(base_num_str))

        next_number = max(numbers) + 1 if numbers else 1
    except (ValueError, FileNotFoundError):
        next_number = 1

    return f"{next_number}.png"

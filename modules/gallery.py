"""sd.cpp-webui - Gallery module"""

import os
import re
import shutil
import subprocess
from typing import List, Tuple, Any, Optional

import gradio as gr

from modules.shared_instance import config
from modules.utils.image_utils import size_extractor
from modules.utils.video_utils import get_avi_resolution
from modules.utils.metadata_utils import (
    parse_png_metadata, parse_jpg_metadata,
    extract_params_from_text
)


PAGE_SIZE = 16

# Output directory of each gallery (ctrl value)
MEDIA_DIRS = [
    'txt2img_dir', 'img2img_dir', 'imgedit_dir',
    'any2video_dir', 'upscale_dir'
]
MEDIA_EXTENSIONS = (
    '.png', '.jpg', '.jpeg', '.webp', '.avi', '.webm', '.mp4'
)


def _natural_key(path: str) -> list:
    """Sort key that orders "2.png" before "10.png"."""
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r'(\d+)', os.path.basename(path).lower())
    ]


def _ctime(path: str) -> float:
    """Creation time, files removed while sorting go last."""
    try:
        return os.path.getctime(path)
    except OSError:
        return float('inf')


class GalleryManager:
    """Controls the gallery block"""

    def __init__(self):
        self.page_num: int = 1
        self.ctrl: int = 0

        self.sort_order: str = config.get('def_gallery_sorting', 'Date (Oldest First)')

        # Files shown by the last reload, selections refer to these
        self.files: List[str] = []

        self.selected_media_index_on_page: Optional[int] = None
        self.selected_media_global_index: Optional[int] = None
        self.current_media_path: Optional[str] = None

    def _get_current_dir(self) -> str:
        """
        Determines the directory based on the control value.
        Read from the config every time, so a changed or newly created
        output directory is picked up.
        """
        if 0 <= self.ctrl < len(MEDIA_DIRS):
            return config.get_or_create_dir(MEDIA_DIRS[self.ctrl])
        return config.get_or_create_dir(MEDIA_DIRS[0])

    def _get_sorted_files(self) -> List[str]:
        """
        Gets all image files from the current directory,
        sorted by creation time.
        """
        media_dir = self._get_current_dir()
        try:
            files = [
                os.path.join(media_dir, f)
                for f in os.listdir(media_dir)
                if f.lower().endswith(MEDIA_EXTENSIONS)
            ]
        except (FileNotFoundError, OSError):
            return []

        if self.sort_order == "Date (Newest First)":
            return sorted(files, key=_ctime, reverse=True)
        elif self.sort_order == "Name (A-Z)":
            return sorted(files, key=_natural_key)
        elif self.sort_order == "Name (Z-A)":
            return sorted(files, key=_natural_key, reverse=True)
        else:
            # Default: Date (Oldest First)
            return sorted(files, key=_ctime)

    @staticmethod
    def _total_pages(total_items: int) -> int:
        """Number of gallery pages, at least one."""
        return (total_items + PAGE_SIZE - 1) // PAGE_SIZE or 1

    def reload_gallery(
        self, page_num: int = 1, ctrl_inp: Optional[int] = None,
        sort_inp: Optional[str] = None, selected_index: Optional[int] = None
    ) -> Tuple[gr.update, int]:
        """Reloads the gallery block to a specific page."""

        if sort_inp is not None:
            self.sort_order = sort_inp

        if ctrl_inp is not None:
            self.ctrl = int(ctrl_inp)

        files = self._get_sorted_files()
        self.files = files
        total_pages = self._total_pages(len(files))

        try:
            page_num = int(page_num)
        except (ValueError, TypeError):
            page_num = 1

        if page_num > total_pages:
            page_num = total_pages
        elif page_num < 1:
            page_num = 1

        self.page_num = int(page_num)

        start_index = (self.page_num - 1) * PAGE_SIZE
        end_index = start_index + PAGE_SIZE

        page_files = files[start_index:end_index]

        dir_map = {
            0: 'txt2img',
            1: 'img2img',
            2: 'imgedit',
            3: 'any2video',
            4: 'upscale'
        }
        current_label = f"{dir_map.get(self.ctrl, 'Gallery')} - {self.sort_order}"

        # Reset selections when reloading
        if selected_index is None:
            self.selected_media_index_on_page = None
            self.current_media_path = None

        return (
            gr.Gallery(
                value=page_files,
                label=current_label,
                selected_index=selected_index
            ),
            self.page_num
        )

    def _navigate_page(
        self, direction: int
    ) -> Tuple[gr.update, int]:
        """Helper for next/prev/last page navigation."""
        total_pages = self._total_pages(len(self._get_sorted_files()))

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

    def _read_media_info(self) -> Tuple[Any, ...]:
        """Reads and parses generation data from the current media."""
        raw_text = None
        width = None
        height = None
        file_path_lower = self.current_media_path.lower()

        try:
            if file_path_lower.endswith('.png'):
                raw_text = parse_png_metadata(self.current_media_path)
                width, height = size_extractor(self.current_media_path)
            elif file_path_lower.endswith(('.jpg', '.jpeg')):
                raw_text = parse_jpg_metadata(self.current_media_path)
                width, height = size_extractor(self.current_media_path)
            elif file_path_lower.endswith('.webp'):
                width, height = size_extractor(self.current_media_path)
            elif file_path_lower.endswith('.avi'):
                raw_text = ""
                width, height = get_avi_resolution(self.current_media_path)
        except Exception as e:
            print(f"Failed to read metadata for {self.current_media_path}: {e}")

        params = extract_params_from_text(raw_text) if raw_text else {}
        is_avi = file_path_lower.endswith('.avi')
        ffmpeg_available = shutil.which("ffmpeg") is not None

        btn_update = gr.update(visible=is_avi, interactive=ffmpeg_available)

        return (
            params.get('pprompt', ''), params.get('nprompt', ''),
            width, height, params.get('steps', None),
            params.get('sampler', ''), params.get('scheduler', ''),
            params.get('cfg', None), params.get('seed', None),
            self.current_media_path, raw_text or "", btn_update
        )

    def get_media_info(self, sel_data: gr.SelectData) -> Tuple[Any, ...]:
        """Reads and parses generation data from a selected media."""
        if sel_data is None:
            # Return empty values if no image is selected
            return (
                "", "", None, None, None, "", "", None, None, "", "",
                gr.update(visible=False)
            )

        self.selected_media_index_on_page = sel_data.index
        # Calculate the global index across all pages
        self.selected_media_global_index = (
            ((self.page_num - 1) * PAGE_SIZE) + sel_data.index
        )

        # Use the files shown in the gallery, a new output would shift
        # the indices of a fresh listing
        if self.selected_media_global_index >= len(self.files):
            return (
                "", "", None, None, None, "", "", None, None,
                "", "Image index out of range.", gr.update(visible=False)
            )

        self.current_media_path = self.files[self.selected_media_global_index]

        return self._read_media_info()

    def convert_to_mp4(self) -> Tuple[gr.update, int]:
        """Converts the currently selected .avi file to an .mp4 file."""
        if not self.current_media_path or not self.current_media_path.lower().endswith('.avi'):
            return self.reload_gallery(page_num=self.page_num)

        out_path = os.path.splitext(self.current_media_path)[0] + '.mp4'

        if not os.path.exists(out_path):
            cmd = [
                'ffmpeg', '-y', '-i', self.current_media_path,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-movflags', '+faststart', out_path
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                print(f"\nSuccessfully converted to {out_path}\n")
            except subprocess.CalledProcessError as e:
                print(f"\nFFmpeg conversion failed: {e.stderr.decode()}\n")

        # Reload the gallery to surface the new MP4 file
        return self.reload_gallery(page_num=self.page_num)

    def delete_media(self) -> Tuple[Any, ...]:
        """
        Deletes the currently selected media. It then selects the previous
        media, or the next if the first was deleted. If the gallery becomes
        empty, it resets the selection.
        """
        # If no image is selected or the index is unknown, just refresh.
        if (not self.current_media_path or
                not os.path.exists(self.current_media_path) or
                self.selected_media_global_index is None):
            gallery_update, page_num = self.reload_gallery(
                page_num=self.page_num
            )

            return (
                gallery_update, page_num,
                "", "", None, None, None, "", "", None, None,
                "", "", gr.update(visible=False)
            )

        index_to_delete = self.selected_media_global_index
        path_to_delete = self.current_media_path

        try:
            os.remove(path_to_delete)
            print(f"\nDeleted {path_to_delete}")
        except OSError as e:
            print(f"\nError deleting file: {e}")

        files_after_delete = self._get_sorted_files()

        if not files_after_delete:
            self.current_media_path = None
            self.selected_media_index_on_page = None
            self.selected_media_global_index = None

            gallery_update, page_num = self.reload_gallery(page_num=1)

            return (
                gallery_update, page_num,
                "", "", None, None, None, "", "", None, None,
                "", "", gr.update(visible=False)
            )

        new_global_index = index_to_delete - 1 if index_to_delete > 0 else 0

        if new_global_index >= len(files_after_delete):
            new_global_index = len(files_after_delete) - 1

        new_page_num = (new_global_index // PAGE_SIZE) + 1
        new_page_index = new_global_index % PAGE_SIZE

        gallery_update, page_num = self.reload_gallery(
            page_num=new_page_num, selected_index=new_page_index
        )

        self.selected_media_global_index = new_global_index
        self.selected_media_index_on_page = new_page_index
        self.current_media_path = self.files[
            min(new_global_index, len(self.files) - 1)
        ]

        return (gallery_update, page_num, *self._read_media_info())


def get_next_media(subctrl: int) -> str:
    """Creates a new, sequential media name (e.g., '123.png')."""
    if 0 <= subctrl < len(MEDIA_DIRS):
        dir_key = MEDIA_DIRS[subctrl]
    else:
        dir_key = MEDIA_DIRS[0]
    media_out_dir = config.get_or_create_dir(dir_key)

    try:
        numbers = []
        for f in os.listdir(media_out_dir):
            if f.lower().endswith(MEDIA_EXTENSIONS):
                name_without_ext = os.path.splitext(f)[0]

                base_num_str = name_without_ext.split('_')[0]

                if base_num_str.isdigit():
                    numbers.append(int(base_num_str))

        next_number = max(numbers) + 1 if numbers else 1
    except (ValueError, FileNotFoundError):
        next_number = 1

    extension = ".webm" if subctrl == 3 else ".png"

    return f"{next_number}{extension}"

"""sd.cpp-webui - Gallery module"""

import os
import re
from PIL import Image

import gradio as gr

from modules.config import (
    txt2img_dir, img2img_dir
)


class GalleryManager:
    """Controls the gallery block"""
    def __init__(self, txt2img_gallery, img2img_gallery):
        self.page_num = 1
        self.ctrl = 0
        self.txt2img_dir = txt2img_gallery
        self.img2img_dir = img2img_gallery
        self.img_index = int
        self.sel_img = int
        self.img_path = str
        self.exif = str

    def _get_img_dir(self):
        """Determines the directory based on the control value"""
        if self.ctrl == 0:
            return self.txt2img_dir
        if self.ctrl == 1:
            return self.img2img_dir
        return self.txt2img_dir

    def reload_gallery(self, ctrl_inp=None, fpage_num=1, subctrl=0):
        """Reloads the gallery block"""
        if ctrl_inp is not None:
            self.ctrl = int(ctrl_inp)
        img_dir = self._get_img_dir()
        # Use a generator to find image files, avoiding the creation of a full list
        def image_files_gen(directory):
            for file in os.listdir(directory):
                if file.endswith(('.jpg', '.png')):
                    yield file

        # Get start and end indices based on the current page
        start_index = (fpage_num * 16) - 16
        end_index = start_index + 16

        # Process image files lazily
        imgs = []
        for i, file_name in enumerate(image_files_gen(img_dir)):
            if start_index <= i < end_index:
                image_path = os.path.join(img_dir, file_name)
                image = Image.open(image_path)
                imgs.append(image)
            elif i >= end_index:
                break
        self.page_num = fpage_num
        if subctrl == 0:
            return imgs, self.page_num, gr.Gallery(selected_index=None)
        return imgs

    def goto_gallery(self, fpage_num=1):
        """Loads a specific gallery page"""
        img_dir = self._get_img_dir()
        files = os.listdir(img_dir)
        total_imgs = len([file for file in files
                         if file.endswith(('.png', '.jpg'))])
        total_pages = (total_imgs + 15) // 16
        if fpage_num is None:
            fpage_num = 1
        self.page_num = min(fpage_num, total_pages)
        return self.reload_gallery(self.ctrl, self.page_num, subctrl=0)

    def next_page(self):
        """Moves to the next gallery page"""
        next_page_num = self.page_num + 1
        img_dir = self._get_img_dir()
        files = os.listdir(img_dir)
        total_imgs = len([file for file in files
                         if file.endswith(('.png', '.jpg'))])
        total_pages = (total_imgs + 15) // 16
        if next_page_num > total_pages:
            self.page_num = 1
        else:
            self.page_num = next_page_num
        imgs = self.reload_gallery(self.ctrl, self.page_num, subctrl=1)
        return imgs, self.page_num, gr.Gallery(selected_index=None)

    def prev_page(self):
        """Moves to the previous gallery page"""
        prev_page_num = self.page_num - 1
        img_dir = self._get_img_dir()
        files = os.listdir(img_dir)
        total_imgs = len([file for file in files
                         if file.endswith(('.png', '.jpg'))])
        total_pages = (total_imgs + 15) // 16
        if prev_page_num < 1:
            self.page_num = total_pages
        else:
            self.page_num = prev_page_num
        imgs = self.reload_gallery(self.ctrl, self.page_num, subctrl=1)
        return imgs, self.page_num, gr.Gallery(selected_index=None)

    def extract_exif_from_jpg(self, img_path):
        """Extracts exif data from jpg"""
        img = Image.open(img_path)
        exif_data = img._getexif()

        if exif_data is not None:
            user_comment = exif_data.get(37510)  # 37510 = UserComment tag
            if user_comment:
                self.exif = f"JPG: Exif\nPositive prompt: "\
                            f"{user_comment.decode('utf-8')[9::2]}"
                return self.exif

            return "JPG: No User Comment found."

        return "JPG: Exif\nNo EXIF data found."

    def last_page(self):
        """Moves to the last gallery page"""
        img_dir = self._get_img_dir()
        files = os.listdir(img_dir)
        total_imgs = len([file for file in files
                         if file.endswith(('.png', '.jpg'))])
        total_pages = (total_imgs + 15) // 16
        self.page_num = total_pages
        imgs = self.reload_gallery(self.ctrl, self.page_num, subctrl=1)
        return imgs, self.page_num, gr.Gallery(selected_index=None)

    def img_info(self, sel_img: gr.SelectData):
        """Reads generation data from an image"""
        if hasattr(sel_img, 'index'):
            self.img_index = (self.page_num * 16) - 16 + sel_img.index
            self.sel_img = sel_img.index
        else:
            self.img_index = (self.page_num * 16) - 16 + self.sel_img
        img_dir = self._get_img_dir()
         # Use a generator to find and sort image files on demand
        def image_file_gen(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg')):
                    yield file_path

        # Sort files only when necessary and only the files we need
        file_paths = sorted(image_file_gen(img_dir), key=os.path.getctime)

        # Initialize parameters
        pprompt = ""
        nprompt = ""
        steps = ""
        sampler = ""
        seed = ""
        exif = ""

        # Handle index out of range errors
        try:
            self.img_path = file_paths[self.img_index]
        except IndexError:
            return "Image index is out of range."
        if self.img_path.endswith(('.jpg', '.jpeg')):
            im = Image.open(self.img_path)
            w, h = im.size
            width = w
            height = h
            exif = self.extract_exif_from_jpg(self.img_path)
            return pprompt, nprompt, width, height, steps, sampler, seed, exif, self.img_path
        if self.img_path.endswith('.png'):
            with open(self.img_path, 'rb') as file:
                if file.read(8) != b'\x89PNG\r\n\x1a\n':
                    im = Image.open(self.img_path)
                    w, h = im.size
                    width = w
                    height = h
                    return pprompt, nprompt, width, height, steps, sampler, seed, self.img_path, exif
                while True:
                    length_chunk = file.read(4)
                    if not length_chunk:
                        im = Image.open(self.img_path)
                        w, h = im.size
                        width = w
                        height = h
                        return pprompt, nprompt, width, height, steps, sampler, seed, self.img_path, exif
                    length = int.from_bytes(length_chunk, byteorder='big')
                    chunk_type = file.read(4).decode('utf-8')
                    png_block = file.read(length)
                    _ = file.read(4)
                    if chunk_type == 'tEXt':
                        _, value = png_block.split(b'\x00', 1)
                        png_exif = f"{value.decode('utf-8')}"

                        if '{"text":' in png_exif:
                            # JSON-like metadata branch.
                            exif = png_exif
                            ppattern = r'.*?{"text":\s*"([^"]*)",\s*"clip".*'
                            npattern = r'(?=.*{"text":\s*"([^"]*)",\s*"clip".*)(.*{"text":\s*"([^"]*)",\s*"clip".*)'
                        else:
                            # Already formatted metadata branch. Since "Positive prompt:" is added later,
                            # we prepend it here.
                            exif = f"PNG: tEXt\nPositive prompt: {png_exif}"
                            # This regex captures the positive prompt whether or not it is enclosed in quotes.
                            ppattern = r'Positive prompt:\s*(?:"(?P<quoted_pprompt>(?:\\.|[^"\\])*)"|(?P<unquoted_pprompt>.*?))\s*(?=\s*(?:Steps:|Negative prompt:))'
                            npattern = r'Negative prompt:\s*(?:"(?P<quoted_nprompt>(?:\\.|[^"\\])*)"|(?P<unquoted_nprompt>.*?))\s*(?=\s*Steps:)'


                        pmatch = re.search(ppattern, exif)
                        nmatch = re.search(npattern, exif)

                        if pmatch and pmatch.lastindex is not None:
                            pprompt = pmatch.group("quoted_pprompt") if pmatch.group("quoted_pprompt") is not None else pmatch.group("unquoted_pprompt")

                        if nmatch and nmatch.lastindex is not None:
                            nprompt = nmatch.group("quoted_nprompt") if nmatch.group("quoted_nprompt") is not None else nmatch.group("unquoted_nprompt")

                        size_pattern = r'Size:\s*(\d+)\s*[xX]\s*(\d+)(?!.*Size:)'
                        size_match = re.search(size_pattern, exif)
                        if size_match:
                            width = size_match.group(2)
                            height = size_match.group(1)
                        else:
                            im = Image.open(self.img_path)
                            w, h = im.size
                            width = w
                            height = h

                        steps_pattern = r'Steps:\s*(\d+)(?!.*Steps:)'
                        steps_match = re.search(steps_pattern, exif, re.DOTALL)
                        if steps_match:
                            steps = steps_match.group(1)

                        sampler_pattern = r'Sampler:\s*([^\s,]+)(?!.*Sampler:)'
                        sampler_match = re.search(sampler_pattern, exif, re.DOTALL)
                        if sampler_match:
                            sampler = sampler_match.group(1)

                        seed_pattern = r'Seed:\s*(\d+)(?!.*Seed:)'
                        seed_match = re.search(seed_pattern, exif, re.DOTALL)
                        if seed_match:
                            seed = int(seed_match.group(1))

                        return pprompt, nprompt, width, height, steps, sampler, seed, self.img_path, exif
        return None

    def delete_img(self):
        """Deletes a selected image"""
        try:
            os.remove(self.img_path)
            print(f"Deleted {self.img_path}")
            self.img_index -= 1
            img_dir = self._get_img_dir()
            files = os.listdir(img_dir)
            total_imgs = len([file for file in files
                             if file.endswith(('.png', '.jpg'))])
            file_paths = [os.path.join(img_dir, file)
                          for file in files
                          if os.path.isfile(os.path.join(img_dir, file)) and
                          file.lower().endswith(('.png', '.jpg'))]
            file_paths.sort(key=os.path.getctime)
            if total_imgs == 0:
                self.sel_img = None
            if self.img_index == total_imgs:
                if self.sel_img == 0 or self.sel_img % 16 == 0:
                    self.sel_img = 16
                    self.page_num -= 1

                else:
                    self.sel_img -= 1

            try:
                self.img_path = file_paths[self.img_index]
            except IndexError:
                return "Image index is out of range."

            imgs, _, _ = self.reload_gallery(None, self.page_num)
            img_info = self.img_info(self.sel_img)
            pprompt_out, nprompt_out, exif = img_info[:3]
            return [imgs, self.page_num, gr.update(self.sel_img),
                    pprompt_out, nprompt_out, exif]
        except FileNotFoundError as e:
            print(f"Error deleting image: {e}")
            return "An error occurred while deleting."


def get_next_img(subctrl):
    """Creates a new image name"""
    if subctrl == 0:
        fimg_out = txt2img_dir
    elif subctrl == 1:
        fimg_out = img2img_dir
    else:
        fimg_out = txt2img_dir
    files = os.listdir(fimg_out)
    png_files = [file for file in files if file.endswith('.png') and
                 file[:-4].isdigit()]
    if not png_files:
        return "1.png"
    highest_number = max(int(file[:-4]) for file in png_files)
    next_number = highest_number + 1
    fnext_img = f"{next_number}.png"
    return fnext_img

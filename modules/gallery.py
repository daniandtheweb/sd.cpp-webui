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
        if fpage_num == None or fpage_num<1:
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
        steps = None
        sampler = ""
        cfg = None
        seed = None
        exif = ""
        width = None
        height = None

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
            return pprompt, nprompt, width, height, steps, sampler, cfg, seed, exif, self.img_path
        if self.img_path.endswith('.png'):
            with open(self.img_path, 'rb') as file:
                if file.read(8) != b'\x89PNG\r\n\x1a\n':
                    im = Image.open(self.img_path)
                    w, h = im.size
                    width = w
                    height = h
                    return pprompt, nprompt, width, height, steps, sampler, cfg, seed, self.img_path, exif
                while True:
                    length_chunk = file.read(4)
                    if not length_chunk:
                        im = Image.open(self.img_path)
                        w, h = im.size
                        width = w
                        height = h
                        return pprompt, nprompt, width, height, steps, sampler, cfg, seed, self.img_path, exif
                    length = int.from_bytes(length_chunk, byteorder='big')
                    chunk_type = file.read(4).decode('utf-8')
                    png_block = file.read(length)
                    _ = file.read(4)
                    if chunk_type == 'tEXt':
                        _, value = png_block.split(b'\x00', 1)
                        png_exif = f"{value.decode('utf-8')}"

                        # Main parsing method
                        exif = f"PNG: tEXt\nPositive prompt: {png_exif}"
                        ppattern = r'Positive prompt:\s*(?:"(?P<quoted_pprompt>(?:\\.|[^"\\])*)"|(?P<unquoted_pprompt>.*?))\s*(?=\s*(?:Steps:|Negative prompt:))'
                        npattern = r'Negative prompt:\s*(?:"(?P<quoted_nprompt>(?:\\.|[^"\\])*)"|(?P<unquoted_nprompt>.*?))\s*(?=\s*Steps:)'

                        pmatch = re.search(ppattern, exif)
                        nmatch = re.search(npattern, exif)

                        if pmatch:
                            pprompt = pmatch.group("quoted_pprompt") or pmatch.group("unquoted_pprompt") or ""
                        if nmatch:
                            nprompt = nmatch.group("quoted_nprompt") or nmatch.group("unquoted_nprompt") or ""

                        # Fallback parsing method (ComfyUI format)
                        if not pprompt and '{"text":' in png_exif:
                            exif = png_exif
                            pattern = r'"text":\s*"((?:[^"\\]|\\.)*)"'
                            matches = re.findall(pattern, exif)

                            if len(matches) > 0:
                                pprompt = matches[0]
                            if len(matches) > 1:
                                nprompt = matches[1]

                            steps_match_json = re.search(r'"steps":\s*(\d+)', exif)
                            if steps_match_json:
                                steps = int(steps_match_json.group(1))

                            sampler_match_json = re.search(r'"sampler_name":\s*"([^"]+)"', exif)
                            if sampler_match_json:
                                sampler = sampler_match_json.group(1)

                            cfg_match_json = re.search(r'"cfg":\s*(\d+)', exif)
                            if cfg_match_json:
                                cfg = float(int(cfg_match_json.group(1)))

                            seed_match_json = re.search(r'"seed":\s*(\d+)', exif)
                            if seed_match_json:
                                seed = int(seed_match_json.group(1))

                        im = Image.open(self.img_path)
                        w, h = im.size
                        width = w
                        height = h

                        if not steps:
                            steps_pattern = r'Steps:\s*(\d+)(?!.*Steps:)'
                            steps_match = re.search(steps_pattern, exif, re.DOTALL)
                            if steps_match:
                                steps = int(steps_match.group(1))
                            else:
                                steps = None

                        if not sampler:
                            sampler_pattern = r'Sampler:\s*([^\s,]+)(?!.*Sampler:)'
                            sampler_match = re.search(sampler_pattern, exif, re.DOTALL)
                            if sampler_match:
                                sampler = sampler_match.group(1)
                            else:
                                sampler = ""

                        if not cfg:
                            cfg_pattern = r'CFG scale:\s*(\d+)(?!.*CFG scale:)'
                            cfg_match = re.search(cfg_pattern, exif, re.DOTALL)
                            if cfg_match:
                                cfg = float(cfg_match.group(1))
                            else:
                                cfg = None

                        if not seed:
                            seed_pattern = r'Seed:\s*(\d+)(?!.*Seed:)'
                            seed_match = re.search(seed_pattern, exif, re.DOTALL)
                            if seed_match:
                                seed = int(seed_match.group(1))
                            else:
                                seed = None

                        return pprompt, nprompt, width, height, steps, sampler, cfg, seed, self.img_path, exif
        return None

    def delete_img(self):
        """Deletes a selected image"""
        try:
            if not hasattr(self, 'img_path') or not os.path.exists(self.img_path):
                print("Deletion failed: No valid image selected or file does not exist.")
                imgs, page_num, gallery_update = self.reload_gallery(self.ctrl, self.page_num, subctrl=0)
                (pprompt, nprompt, width, height, steps, sampler, cfg, seed, img_path, exif) = ("", "", None, None, None, "", None, None, "", "")
                return (imgs, page_num, gallery_update, pprompt, nprompt, width, height, steps, sampler, cfg, seed, img_path, exif)

            index_of_deleted_img = self.img_index

            os.remove(self.img_path)
            print(f"Deleted {self.img_path}")
            img_dir = self._get_img_dir()
            file_paths = sorted(
                [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))],
                key=os.path.getctime
            )
            total_imgs = len(file_paths)

            if total_imgs == 0:
                self.page_num, self.sel_img, self.img_index, self.img_path = 1, None, 0, ""
                return ([], 1, gr.Gallery(value=None, selected_index=None),
                        "", "", None, None, None, "", None, None, "", "")

            new_selected_index = index_of_deleted_img
            if new_selected_index >= total_imgs:
                new_selected_index = total_imgs - 1
            new_selected_index = max(0, new_selected_index)

            self.page_num = (new_selected_index // 16) + 1
            self.sel_img = new_selected_index % 16

            imgs, _, _ = self.reload_gallery(self.ctrl, self.page_num, subctrl=0)

            img_info_tuple = self.img_info(self.sel_img)
            if img_info_tuple is None:
                raise ValueError("Failed to retrieve information for the new image.")

            pprompt, nprompt, width, height, steps, sampler, cfg, seed, img_path, exif = img_info_tuple
            gallery_update = gr.Gallery(selected_index=self.sel_img)

            return (
                imgs, self.page_num, gallery_update,
                pprompt, nprompt, width, height, steps, sampler, cfg, seed, img_path, exif
            )

        except Exception as e:
            print(f"An error occurred in delete_img: {e}")
            return ([], 1, gr.Gallery(value=None), "", "", None, None, None, "", None, None, "", "")


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

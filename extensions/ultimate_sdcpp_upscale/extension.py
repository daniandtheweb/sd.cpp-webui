"""Ultimate SDCPP Upscale: upscale the images a generation makes, tile by tile.

Turn it on in a generation tab (txt2img, img2img, imgedit) and, after the
tab makes its images, each one is enlarged and redrawn with img2img, a
tile at a time, so it gains detail without the memory of a huge image.
The upscale has its own settings, in the Ultimate SDCPP Upscale tab, like the
ComfyUI node of the same name: the upscale model, the model, prompts,
sampler, scheduler and denoise of the redraw, the tile size, padding, edge
blur and order (linear or chess), seam fixing (band pass or half tile),
uniform tiles, tiled VAE decoding, and a ControlNet (tile) to keep the
redraw close to the image. While it works, the results show the image
with the tile being redrawn lit up, and what the model is drawing in it,
step by step.

The images the tab made are saved, by the tab, before the upscale starts:
they stay where it saves them whatever happens next. If the upscale is
stopped or fails, what it had upscaled so far is saved too (as
<number>_stopped.png or <number>_partial.png) and shown with the original.
The upscaled ones are in the extension's output folder (its own Gallery
button).

Credits: the idea and the tiling and seam fixing methods are not mine.
They come from the ComfyUI node ComfyUI_UltimateSDUpscale, by ssitu
(see COMFYUI_URL), which is based on ultimate-upscale-for-automatic1111,
by Coyote-A (see A1111_URL). This is an independent implementation for
stable-diffusion.cpp.

The redraw is a generation per tile through the app: in server mode the
model stays loaded; in CLI mode sd-cli starts, and loads it, for every
tile, which is much slower. In server mode the server is restarted, and
put back after, if the redraw needs a ControlNet or a model it doesn't have.
"""

import os
import random
import time

from PIL import Image, PngImagePlugin

from . import settings, tiles

NAME = "Ultimate SDCPP Upscale"
VERSION = "1.0"
AUTHOR = "daniandtheweb"
URL = "https://github.com/daniandtheweb/sd.cpp-webui"
DESCRIPTION = (
    "Upscales the images a generation makes and redraws them tile by "
    "tile, with its own model, prompt, sampler and ControlNet settings. "
    "Based on ComfyUI_UltimateSDUpscale and "
    "ultimate-upscale-for-automatic1111."
)
API_VERSION = 1

# The projects this one is based on
COMFYUI_URL = "https://github.com/ssitu/ComfyUI_UltimateSDUpscale"
A1111_URL = "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111"

TABS = ('txt2img', 'img2img', 'imgedit')
RESAMPLING = Image.LANCZOS
# The upscaled size is a multiple of this: the model works on 8 pixels
# (every tile is cut to what it takes, see tiles.py)
ROUND = 8
# The live preview of a tile is drawn at most this often (seconds)
FRAME_EVERY = 0.2


def register(api):
    api.add_output(label="Ultimate SDCPP Upscale")
    api.add_tab("Ultimate SDCPP Upscale", lambda: render_tab(api))
    api.add_ui(render_options, tabs=TABS)
    api.add_hook("replace_generation", lambda ctx: replace(api, ctx))


# ---------------------------------------------------------------------- UI

def render_options(tab):
    import gradio as gr

    return {
        'enabled': gr.Checkbox(
            label="Ultimate SDCPP Upscale",
            info="Upscale the images this tab makes, tile by tile, with "
                 "the settings of the Ultimate SDCPP Upscale tab. The "
                 "images of the tab are kept."
        ),
    }


def render_tab(api):
    import gradio as gr

    cfg = settings.load(api)
    fields = {}
    models = {
        'upscaler': [settings.NO_UPSCALER] + api.models('upscaler'),
        'checkpoint': api.models('checkpoint'),
        'vae': api.models('vae'),
        'unet': api.models('diffusion'),
        'unet_vae': api.models('vae'),
        'clip_l': api.models('text_encoder'),
        'clip_g': api.models('text_encoder'),
        't5xxl': api.models('text_encoder'),
        'llm': api.models('text_encoder'),
        'cn_model': api.models('controlnet'),
    }

    def add(key, component):
        fields[key] = component
        return component

    def number(key, label, minimum, maximum, step, info=None):
        return add(key, gr.Slider(
            label=label, minimum=minimum, maximum=maximum, step=step,
            value=cfg[key], info=info))

    def model(key, label, empty=True, **kwargs):
        choices = models[key] if not empty else [""] + models[key]
        return add(key, gr.Dropdown(
            label=label, choices=choices, value=cfg[key],
            allow_custom_value=True, **kwargs))

    gr.Markdown(
        "## Ultimate SDCPP Upscale\n\n"
        "Settings of the upscale that the *Ultimate SDCPP Upscale* option "
        "of the generation tabs does to the images they make. They are "
        "saved as you change them, and the Save button saves them right "
        "away.\n\n"
        "Based on the ideas of "
        f"[ComfyUI_UltimateSDUpscale]({COMFYUI_URL}) and "
        f"[ultimate-upscale-for-automatic1111]({A1111_URL})."
    )

    with gr.Accordion("Upscale", open=True):
        number('scale', "Scale", 1, 8, 0.25, "How much bigger the image gets.")
        model('upscaler', "Upscale model", empty=False,
              info="Enlarges the image before the redraw. Without one it "
                   "is a plain resize.")

    with gr.Accordion("Redraw model", open=False):
        add('model_from', gr.Radio(
            [settings.SAME, settings.CUSTOM], value=cfg['model_from'],
            label="Model"))
        with gr.Group(visible=cfg['model_from'] == settings.CUSTOM) as custom:
            add('model_type', gr.Radio(
                [settings.CHECKPOINT, settings.UNET], label="Type",
                value=cfg['model_type']))
            with gr.Group(visible=cfg['model_type'] == settings.CHECKPOINT
                          ) as checkpoint:
                model('checkpoint', "Checkpoint")
                model('vae', "VAE")
            with gr.Group(visible=cfg['model_type'] == settings.UNET
                          ) as unet:
                model('unet', "UNET")
                model('unet_vae', "VAE")
                model('clip_l', "clip_l")
                model('clip_g', "clip_g")
                model('t5xxl', "t5xxl")
                model('llm', "LLM")
        reload = gr.Button("Reload the lists of models")

    with gr.Accordion("Prompts", open=False):
        add('prompt', gr.Textbox(
            label="Prompt", lines=2, value=cfg['prompt'],
            info="Empty: the prompt of the image."))
        add('negative', gr.Textbox(
            label="Negative prompt", lines=2, value=cfg['negative'],
            info="Empty: the negative prompt of the image."))

    with gr.Accordion("Sampling", open=False):
        add('sampler', gr.Dropdown(
            [settings.SAME] + api.choices('samplers'), value=cfg['sampler'],
            label="Sampler", allow_custom_value=True))
        add('scheduler', gr.Dropdown(
            [settings.SAME] + api.choices('schedulers'),
            value=cfg['scheduler'], label="Scheduler",
            allow_custom_value=True))
        number('steps', "Steps", 1, 150, 1)
        number('cfg', "CFG scale", 0, 30, 0.5)
        number('denoise', "Denoise", 0, 1, 0.01,
               "How much each tile changes: 0.2 to 0.4 is usual.")
        add('seed', gr.Number(
            label="Seed", value=cfg['seed'], precision=0,
            info="-1: a random one, the same in every tile."))

    with gr.Accordion("Tiles", open=False):
        add('mode', gr.Radio(
            list(tiles.MODES), value=cfg['mode'], label="Tile mode",
            info="Linear: row by row. Chess: one colour of a chessboard, "
                 "then the other. None: only upscale."))
        number('tile_width', "Tile width", 64, 2048, 64)
        number('tile_height', "Tile height", 64, 2048, 64)
        number('padding', "Padding", 0, 512, 8,
               "Pixels around a tile the model sees.")
        number('blur', "Edge blur", 0, 64, 1,
               "How soft the edges of the tiles are when they are joined.")
        add('uniform', gr.Checkbox(
            label="Uniform tiles", value=cfg['uniform'],
            info="Send every tile at the same size, also at the borders."))
        add('tiled_decode', gr.Checkbox(
            label="Tiled VAE decode", value=cfg['tiled_decode'],
            info="Decode each tile in pieces: less memory."))
        add('live_preview', gr.Checkbox(
            label="Live preview of the tiles", value=cfg['live_preview'],
            info="Show what the model draws in each tile while it works, "
                 "with the preview settings of the tab (or a quick one)."))

    with gr.Accordion("Seam fix", open=False):
        add('seam', gr.Dropdown(
            list(tiles.SEAM_MODES), value=cfg['seam'], label="Seam mode",
            info="A second go over the seams between the tiles."))
        number('seam_denoise', "Seam denoise", 0, 1, 0.01)
        number('seam_width', "Seam band width", 16, 512, 8,
               "Of the bands of Band Pass.")
        number('seam_padding', "Seam padding", 0, 512, 8,
               "Pixels around a seam the model sees.")
        number('seam_blur', "Seam edge blur", 0, 64, 1,
               "How soft the edges of the seam fix are.")

    with gr.Accordion("ControlNet", open=False):
        add('controlnet', gr.Checkbox(
            label="Use a ControlNet", value=cfg['controlnet'],
            info="Every tile is its own control image: a tile model keeps "
                 "the redraw close to what is there."))
        model('cn_model', "ControlNet")
        number('cn_strength', "Strength", 0, 2, 0.05)
        add('cn_union', gr.Checkbox(
            label="ControlNet Union", value=cfg['cn_union']))
        add('cn_type', gr.Dropdown(
            api.choices('control_types'), value=cfg['cn_type'],
            label="Control type", allow_custom_value=True,
            info="For a ControlNet Union: tile."))

    keys = list(fields)
    components = [fields[key] for key in keys]
    save = gr.Button("Save", variant="primary")
    reset = gr.Button("Restore the defaults")

    def save_now(*values):
        settings.save(api, dict(zip(keys, values)))
        gr.Info("Ultimate SDCPP Upscale settings saved.")

    gr.on(
        [component.change for component in components],
        lambda *values: settings.save(api, dict(zip(keys, values))),
        inputs=components
    )
    save.click(save_now, inputs=components)
    fields['model_from'].change(
        lambda value: gr.update(visible=value == settings.CUSTOM),
        inputs=[fields['model_from']], outputs=[custom])
    fields['model_type'].change(
        lambda value: (gr.update(visible=value == settings.CHECKPOINT),
                       gr.update(visible=value == settings.UNET)),
        inputs=[fields['model_type']], outputs=[checkpoint, unet])

    def reload_models():
        lists = {
            'upscaler': [settings.NO_UPSCALER] + api.models('upscaler'),
            'checkpoint': api.models('checkpoint'),
            'vae': api.models('vae'), 'unet': api.models('diffusion'),
            'unet_vae': api.models('vae'),
            'clip_l': api.models('text_encoder'),
            'clip_g': api.models('text_encoder'),
            't5xxl': api.models('text_encoder'),
            'llm': api.models('text_encoder'),
            'cn_model': api.models('controlnet'),
        }
        return [gr.update(choices=(
            lists[key] if key == 'upscaler' else [""] + lists[key]))
            for key in lists]

    reload.click(reload_models, outputs=[fields[key] for key in models])

    def restore():
        fresh = settings.defaults(api)
        return [fresh[key] for key in keys]

    reset.click(restore, outputs=components)


# ---------------------------------------------------------------- the work

def replace(api, ctx):
    """The replace_generation hook: take over only when it is turned on."""
    if ctx.tab not in TABS or not api.param(ctx.params, 'enabled', False):
        return None
    return pipeline(api, ctx.tab, ctx.params)


def pipeline(api, tab: str, params: dict):
    """
    Generate as the tab would and upscale each image it makes: yields what
    the app shows (progress and previews) and then a Progress with the
    main bar, the bar of the tile and the preview, and returns the paths
    of the upscaled ones, each with its original. If the upscale is
    stopped or fails, it returns what it had upscaled so far (saved) and
    the saved originals.
    """
    cfg = settings.load(api)
    # The tab's own generation, with its progress and preview. It is saved
    # (keep) by the app before anything else happens
    images = yield from api.generate_live(params, tab=tab, keep=True)
    if not images:
        raise RuntimeError("the generation made no image")
    originals = [getattr(image, 'filename', None) for image in images]
    paths = []
    # In server mode what the tiles load (the ControlNet) is chosen when the
    # server starts: it is restarted with it and put back as it was
    before = yield from api.serve(settings.server_needs(
        settings.tile_params(api, cfg, params, 0)))
    try:
        for number, image in enumerate(images):
            result, seed, size = yield from upscale_image(
                api, cfg, params, image.convert('RGB'), number, len(images))
            paths.append(save_result(
                api, cfg, params, result, seed, size, originals[number]))
        if before is not None:
            yield api.Progress(100, "Putting the server back as it was...")
    except (Exception, SystemExit) as error:        # including Stopped
        stopped = isinstance(error, api.Stopped)
        if not stopped:
            print("Ultimate SDCPP Upscale failed: "
                  f"{type(error).__name__}: {error}")
        left = originals[len(paths):]
        partial = getattr(error, 'partial', None)
        kept = []
        if partial:
            canvas, seed, size, done, total = partial
            kept.append(save_result(
                api, cfg, params, canvas, seed, size, left[0] if left else
                None, "_stopped" if stopped else "_partial",
                f"stopped after {done} of {total} areas" if stopped else
                f"failed after {done} of {total} areas"))
        saved = [o for o in left if o]
        names = ", ".join(os.path.basename(p) for p in kept + saved)
        what = "Stopped" if stopped else f"The upscale failed ({error})"
        yield api.Progress(
            100, f"{what}: what it had upscaled and the original are saved: "
            f"{names}" if kept else f"{what}: the original is saved: {names}")
        return paths + kept + saved
    finally:
        if before is not None:
            api.unserve(before)
    # Each upscale with the image it comes from, to compare them
    return [path for pair in zip(paths, originals) for path in pair if path]


def enlarge(api, cfg: dict, image: Image.Image, size: tuple, work: str):
    """The image at the size of the upscale: by the model, then resized."""
    name = cfg['upscaler']
    if name and name != settings.NO_UPSCALER and size != image.size:
        image.save(work)
        made = api.generate({
            **api.default_params(), 'in_img_inp': work,
            'in_upscl': name, 'in_upscl_rep': 1,
        }, tab="upscale")
        image = made[0].convert('RGB')
    return image if image.size == size else image.resize(size, RESAMPLING)


def preview_of(update) -> Image.Image | None:
    """The image of an update of the app (what the model is drawing)."""
    shown = update[3] if isinstance(update, (tuple, list)) and len(
        update) >= 4 else None
    if not isinstance(shown, list) or not shown:
        return None
    first = shown[0]
    try:
        if isinstance(first, Image.Image):
            return first.copy()
        if isinstance(first, str):
            with Image.open(first) as picture:
                return picture.convert('RGB')
    except OSError:
        return None                 # not written yet, or already removed
    return None


def progress_of(update) -> tuple | None:
    """
    (percent, step, total, status) of an update of the app, if it has
    them. The status is empty while the model loads and has the speed
    while it samples.
    """
    bar = update[1] if isinstance(update, (tuple, list)) and len(
        update) >= 4 else None
    percent = getattr(bar, 'percent', None)
    if percent is None or (isinstance(bar, dict)
                           and bar.get('visible') is False):
        return None
    return (percent, getattr(bar, 'step', None), getattr(bar, 'total', None),
            getattr(bar, 'status', ""))


def tile_bar(step) -> tuple | None:
    """
    The second progress bar, under the main one: (percent, text) of the
    area being redrawn, as the model draws it. None when none is.
    """
    if step.current is None:
        return None
    name = f"{step.stage.name} {step.done + 1}"
    if not step.progress:
        return 0, f"{name} · starting"
    percent, number, total, status = step.progress
    what = "step" if status else "loading"      # sampling has a speed
    steps = f" · {what} {int(number)}/{int(total)}" if number and total else ""
    return percent, f"{name}{steps} · {round(percent)}%"


def tile_line(label: str, step, cfg: dict) -> str:
    """What the terminal shows before the run of each tile."""
    left, top, right, bottom = step.current
    denoise = step.stage.denoise
    # sd.cpp runs the last part of the steps: steps * denoising strength
    run = max(1, int(int(cfg['steps']) * denoise)) if denoise < 1 else int(
        cfg['steps'])
    return (f"[Ultimate SDCPP Upscale] {label}{step.stage.name} "
            f"{step.done + 1} of {step.total} · {right - left}x"
            f"{bottom - top} at ({left}, {top}) · denoise {denoise} · "
            f"{run} of {cfg['steps']} steps run")


def upscale_image(api, cfg: dict, image_params: dict, image: Image.Image,
                  number: int, count: int):
    """
    Upscale one image: a generator that yields Progress and returns the
    result, the seed and the size.
    """
    def percent(fraction: float) -> int:
        return round((number + fraction) / count * 100)

    label = f"Image {number + 1} of {count}: " if count > 1 else ""
    scale = float(cfg['scale'])
    size = tuple(
        max(ROUND, round(length * scale / ROUND) * ROUND)
        for length in image.size
    )
    seed = int(cfg['seed'])
    if seed < 0:
        seed = random.randrange(2 ** 32)   # the same one in every tile

    work = os.path.join(api.data_dir(), "tile.png")
    canvas, counts = None, (0, 0)
    try:
        upscaler = cfg['upscaler'] != settings.NO_UPSCALER and cfg[
            'upscaler']
        how = f" with {upscaler}..." if upscaler else "..."
        yield api.Progress(percent(0), f"{label}Upscaling{how}", image)
        base = enlarge(api, cfg, image, size, work)
        canvas = base               # what is saved if the work is cut short
        common = settings.tile_params(api, cfg, image_params, seed)
        control = bool(common.get('in_cnnet_bool'))

        def redraw(crop: Image.Image, denoise: float):
            """A tile redrawn: yields what the model is drawing meanwhile."""
            crop.save(work)
            tile = {**common, 'in_img_inp': work, 'in_strength': denoise,
                    'in_width': crop.width, 'in_height': crop.height}
            if control:
                tile['in_control_img'] = work
            drawing = api.generate_live(tile)
            while True:
                try:
                    update = next(drawing)
                except StopIteration as finished:
                    made = finished.value
                    break
                yield preview_of(update), progress_of(update)
            if not made:
                raise RuntimeError("a tile came back without an image")
            return made[0]

        steps = tiles.process(base, redraw, settings.stages(cfg, size))
        last = 0.0
        while True:
            try:
                step = next(steps)
            except StopIteration as finished:
                result = finished.value
                break
            canvas, counts = step.canvas, (step.done, step.total)
            if step.current is not None and step.live is None:
                print(tile_line(label, step, cfg), flush=True)  # a new one
            text = ("Done" if step.current is None else
                    f"{step.stage.name}: {step.done + 1} of {step.total}")
            fraction = step.done / step.total if step.total else 1
            frame = None
            fresh = step.live is None or step.live[0] is not None
            if fresh and (step.live is None
                          or time.monotonic() - last >= FRAME_EVERY):
                last = time.monotonic()
                frame = tiles.preview(step)     # not every step needs one
            yield api.Progress(
                percent(fraction), label + text, frame, tile_bar(step))
    except (Exception, SystemExit) as error:
        if canvas is not None:      # the pipeline saves it, whatever it is
            error.partial = (canvas.copy(), seed, size, *counts)
        raise
    finally:
        if os.path.exists(work):
            os.remove(work)
    return result, seed, size


def source_text(path: str | None) -> str:
    """The generation info saved in the image that was upscaled, if any."""
    try:
        with Image.open(path) as picture:
            return str(picture.info.get('parameters') or "")
    except (OSError, TypeError, AttributeError):
        return ""


def save_result(api, cfg: dict, params: dict, result: Image.Image,
                seed: int, size: tuple, original: str | None = None,
                suffix: str = "", note: str = "") -> str:
    """
    Save the upscaled image, with its generation info. A suffix
    (_stopped) marks one that is not finished, and note says how far.
    """
    path = api.next_output_path()
    if suffix:
        base, extension = os.path.splitext(path)
        path = f"{base}{suffix}{extension}"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", settings.parameters_text(
        cfg, settings.tile_params(api, cfg, params, seed), size, seed,
        os.path.basename(original) if original else "", note,
        source_text(original)))
    result.save(path, pnginfo=info)
    return path

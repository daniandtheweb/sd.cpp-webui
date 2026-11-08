import os
import random

import pytest
from PIL import Image, ImageChops

# Not a built-in extension (it is not in git): without it, nothing to test
pytest.importorskip('extensions.ultimate_sdcpp_upscale')

from extensions.ultimate_sdcpp_upscale import extension, settings, tiles  # noqa: E402
from modules.extensions.host import (  # noqa: E402
    GenerationStopped, Progress)
from modules.ui.progress import progress_update  # noqa: E402


def run(generator):
    """Drive a generator to its end: (what it yielded, what it returned)."""
    yielded = []
    while True:
        try:
            yielded.append(next(generator))
        except StopIteration as finished:
            return yielded, finished.value


def plan_for(size, **changes):
    options = dict(
        mode=tiles.LINEAR, padding=16, blur=8, denoise=0.3, uniform=False,
        seam=tiles.NO_SEAMS, seam_width=64, seam_padding=16, seam_blur=4,
        seam_denoise=0.4)
    options.update(changes)
    return tiles.stages(size, 128, 128, **options)


def redrawn(image, plan, fn=None):
    fn = fn or (lambda crop, denoise: crop)
    return run(tiles.process(image, fn, plan))


# ------------------------------------------------------------------ tiles

@pytest.mark.parametrize('length, tile', [(1, 512), (63, 256), (1000, 512)])
def test_the_cuts_cover_the_whole_length_with_no_gap(length, tile):
    cut = tiles.cuts(length, tile)
    assert cut[0] == 0 and cut[-1] == length and cut == sorted(cut)


def test_chess_does_one_colour_then_the_other():
    rects = tiles.tile_rects((300, 300), 100, 100, tiles.CHESS)
    cells = [(x // 100, y // 100) for x, y, _, _ in rects]
    parities = [(i + j) % 2 for i, j in cells]
    assert parities == sorted(parities) and set(parities) == {0, 1}
    assert sorted(rects) == sorted(tiles.tile_rects((300, 300), 100, 100))


def test_the_seam_modes_cover_the_cuts_of_the_tiles():
    size = (300, 300)                       # 3 x 3 tiles: 2 cuts each way
    assert tiles.seam_rects(size, 100, 100, tiles.NO_SEAMS, 40) == []
    band = tiles.seam_rects(size, 100, 100, tiles.BAND, 40)
    assert len(band) == 12 and all(
        r[2] - r[0] == 40 or r[3] - r[1] == 40 for r in band)
    half = tiles.seam_rects(size, 100, 100, tiles.HALF, 40)
    assert len(half) == 12 and any(r[2] - r[0] == 100 for r in half)
    plus = tiles.seam_rects(size, 100, 100, tiles.HALF_PLUS, 40)
    assert len(plus) == 12 + 4              # and the 4 crossings


def test_the_stages_follow_the_settings():
    names = [s.name for s in plan_for((300, 300))]
    assert names == ["Tiles"]
    both = plan_for((300, 300), seam=tiles.BAND)
    assert [s.name for s in both] == ["Tiles", "Seams"]
    assert both[1].denoise == 0.4 and both[1].padding == 16
    assert [s.name for s in plan_for(
        (300, 300), mode=tiles.NO_TILES, seam=tiles.HALF)] == ["Seams"]
    assert plan_for((300, 300), mode=tiles.NO_TILES) == []


def test_sides_are_multiples_of_what_sd_cpp_takes_as_it_is():
    assert [tiles.side(v) for v in (10, 64, 65, 575)] == [64, 64, 128, 576]


@pytest.mark.parametrize('mode', [tiles.LINEAR, tiles.CHESS])
def test_every_pixel_is_redrawn_borders_included(mode):
    base = Image.new('RGB', (300, 200), (10, 20, 30))
    target = (200, 100, 50)
    _, result = redrawn(
        base, plan_for(base.size, mode=mode, seam=tiles.HALF_PLUS),
        lambda crop, denoise: Image.new('RGB', crop.size, target))
    worst = ImageChops.difference(
        result, Image.new('RGB', result.size, target)).getextrema()
    assert all(high <= 6 for _, high in worst)


def test_a_redraw_that_changes_nothing_changes_nothing():
    base = Image.effect_noise((300, 200), 40).convert('RGB')
    _, result = redrawn(base, plan_for(
        base.size, seam=tiles.HALF_PLUS, uniform=True))
    assert result.tobytes() == base.tobytes()


def test_each_stage_is_redrawn_with_its_own_denoise():
    asked = []
    base = Image.new('RGB', (300, 200))
    redrawn(base, plan_for(base.size, seam=tiles.BAND),
            lambda crop, denoise: asked.append(denoise) or crop)
    assert sorted(set(asked)) == [0.3, 0.4] and asked[0] == 0.3


def test_uniform_tiles_are_all_sent_at_one_size():
    sizes = []
    base = Image.new('RGB', (400, 300))
    redrawn(base, plan_for(base.size, uniform=True),
            lambda crop, denoise: sizes.append(crop.size) or crop)
    assert len(set(sizes)) == 1
    sizes.clear()
    redrawn(base, plan_for(base.size),
            lambda crop, denoise: sizes.append(crop.size) or crop)
    assert len(set(sizes)) > 1


def test_progress_counts_the_areas_of_every_stage():
    base = Image.new('RGB', (300, 200))
    steps, _ = redrawn(base, plan_for(base.size, seam=tiles.HALF))
    assert (steps[0].done, steps[-1].done) == (0, steps[0].total)
    assert steps[0].total > len(tiles.tile_rects(base.size, 128, 128))
    assert steps[-1].current is None and steps[0].stage.name == "Tiles"
    assert steps[-2].stage.name == "Seams"


def test_a_small_crop_is_asked_at_the_smallest_side():
    asked = []
    base = Image.new('RGB', (40, 30))
    redrawn(base, plan_for(base.size, padding=0, blur=0),
            lambda crop, denoise: asked.append(crop.size) or crop)
    assert asked == [(64, 64)]


def test_the_padding_is_at_least_twice_the_blur():
    """A tile's soft edge has to fade into the next tile, not the old image."""
    plan = plan_for((700, 64), padding=0, blur=4, seam_padding=2,
                    seam_blur=8, seam=tiles.BAND)
    assert [s.padding for s in plan] == [8, 16]
    base = Image.new('RGB', (255, 333), (10, 20, 30))
    target = (200, 100, 50)
    _, result = redrawn(
        base, plan_for(base.size, padding=0, blur=32, mode=tiles.CHESS),
        lambda crop, denoise: Image.new('RGB', crop.size, target))
    worst = ImageChops.difference(
        result, Image.new('RGB', base.size, target)).getextrema()
    assert all(high <= 6 for _, high in worst)      # no old pixel is left


def test_the_geometry_holds_for_all_kinds_of_sizes():
    """Tiles, crops and masks over many random images and settings."""
    rng = random.Random(7)
    for _ in range(150):
        w = rng.choice([1, 40, 64, 100, 255, 300, 513, 700, 1023])
        h = rng.choice([1, 50, 64, 99, 200, 333, 768])
        tile_w, tile_h = rng.choice([64, 100, 192, 512]), rng.choice(
            [64, 128, 256, 777])
        plan = tiles.stages(
            (w, h), tile_w, tile_h, mode=rng.choice(
                [tiles.LINEAR, tiles.CHESS]), padding=rng.choice([0, 16, 64]),
            blur=rng.choice([0, 4, 16]), denoise=0.3,
            uniform=rng.choice([True, False]), seam=rng.choice(
                list(tiles.SEAM_MODES)), seam_width=rng.choice([16, 128]),
            seam_padding=rng.choice([0, 32]), seam_blur=rng.choice([0, 8]),
            seam_denoise=0.3)
        where = (w, h, tile_w, tile_h)
        main = next(s for s in plan if s.name == "Tiles")
        # the tiles cover the image exactly once, none bigger than asked
        assert sum((r[2] - r[0]) * (r[3] - r[1]) for r in main.rects) == (
            w * h), where
        assert all(r[2] - r[0] <= tile_w and r[3] - r[1] <= tile_h
                   for r in main.rects), where
        for stage in plan:
            target = None
            if stage.uniform:
                target = (
                    tiles.side(max(r[2] - r[0] for r in stage.rects)
                               + 2 * stage.padding),
                    tiles.side(max(r[3] - r[1] for r in stage.rects)
                               + 2 * stage.padding))
            sizes = set()
            for rect in stage.rects:
                box = tiles._crop_box(rect, (w, h), stage.padding, target)
                # the crop is in the image and has the tile inside
                assert (0 <= box[0] <= rect[0] and rect[2] <= box[2] <= w
                        and 0 <= box[1] <= rect[1]
                        and rect[3] <= box[3] <= h), (where, rect, box)
                # a side the model takes as it is, or the whole side
                for length, whole in ((box[2] - box[0], w),
                                      (box[3] - box[1], h)):
                    assert length % tiles.ALIGN == 0 or length == whole
                sizes.add((box[2] - box[0], box[3] - box[1]))
            if stage.uniform:
                assert len(sizes) == 1, (where, sizes)
        # and a redraw leaves no old pixel behind
        base = Image.new('RGB', (w, h), (10, 20, 30))
        _, result = redrawn(
            base, plan, lambda crop, denoise: Image.new(
                'RGB', crop.size, (200, 100, 50)))
        worst = ImageChops.difference(
            result, Image.new('RGB', (w, h), (200, 100, 50))).getextrema()
        assert all(high <= 6 for _, high in worst), where


# ---------------------------------------------------------------- preview

def _steps():
    base = Image.new('RGB', (400, 300), (100, 150, 200))
    return redrawn(base, plan_for(base.size, seam=tiles.BAND))[0]


def test_the_preview_dims_what_is_not_redrawn_and_marks_the_current_tile():
    steps = _steps()
    first = steps[0]
    frame = tiles.preview(first)
    assert frame.size == first.canvas.size
    right, top = first.current[2], first.current[1]
    assert sum(frame.getpixel((300, 250))) < sum(first.canvas.getpixel(
        (300, 250))) * 0.7
    assert frame.getpixel((right, top + 40)) != first.canvas.getpixel(
        (right, top + 40))
    assert tiles.preview(steps[-1]).tobytes() == steps[-1].canvas.tobytes()


def test_seams_are_not_dimmed_and_only_the_current_area_is_marked():
    steps = _steps()
    seams = next(s for s in steps if s.stage and s.stage.name == "Seams")
    frame = tiles.preview(seams)
    assert frame.getpixel((10, 10)) == seams.canvas.getpixel((10, 10))
    later = steps[4]
    x0, y0, x1, y1 = steps[3].current       # an area done before: no trail
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    assert tiles.preview(later).getpixel(center) == later.canvas.getpixel(
        center)


def test_a_tile_that_yields_previews_is_seen_while_it_is_drawn():
    base = Image.new('RGB', (300, 200), (30, 60, 90))

    def drawing(crop, denoise):
        yield Image.new('RGB', (32, 32), (250, 0, 0))
        return crop

    steps, _ = redrawn(base, plan_for(base.size), drawing)
    live = [s for s in steps if s.live]
    assert len(live) == len(tiles.tile_rects(base.size, 128, 128))
    step = live[0]
    left, top, right, bottom = step.current
    frame = tiles.preview(step)
    inside = frame.getpixel(((left + right) // 2, (top + bottom) // 2))
    assert inside[0] > 150 and inside[2] < 100      # the live red, not dimmed
    plain = tiles.preview(steps[0])
    assert plain.getpixel(((left + right) // 2, (top + bottom) // 2))[0] < 100


def test_a_tile_can_report_its_progress_without_a_new_picture():
    base = Image.new('RGB', (300, 200), (30, 60, 90))

    def drawing(crop, denoise):
        yield None, (10, 2, 20, "")
        yield Image.new('RGB', (32, 32), (250, 0, 0)), (60, 12, 20, "x")
        return crop

    steps, _ = redrawn(base, plan_for(base.size), drawing)
    live = [s for s in steps if s.live]
    assert live[0].progress == (10, 2, 20, "") and live[0].live[0] is None
    assert live[1].progress == (60, 12, 20, "x")
    assert live[1].live[0] is not None
    left, top, right, bottom = live[0].current
    middle = ((left + right) // 2, (top + bottom) // 2)
    # nothing drawn yet: the tile is still dimmed, like the rest
    assert tiles.preview(live[0]).getpixel(middle)[0] < 100
    assert tiles.preview(live[1]).getpixel(middle)[0] > 150


def test_a_big_image_is_shown_small():
    base = Image.new('RGB', (4096, 2048))
    step = next(tiles.process(base, lambda c, d: c,
                              plan_for(base.size)))
    assert max(tiles.preview(step).size) == tiles.PREVIEW_SIDE


# --------------------------------------------------------------- settings

class FakeApi:
    Stopped = GenerationStopped
    Progress = Progress

    def __init__(self, tmp_path):
        self.tmp, self.runs, self.stop_at = tmp_path, [], None
        self.upscaler, self.fail_at = True, None
        self.batch, self.saved = 1, 0
        self.served, self.unserved, self.before = [], [], None

    def serve(self, changes):
        self.served.append(changes)
        return self.before
        yield                       # a generator, like the app's

    def unserve(self, before):
        self.unserved.append(before)

    def param(self, params, name, default=None):
        return params.get(f'ext_ultimate_sdcpp_upscale_{name}', default)

    def data_dir(self):
        return str(self.tmp)

    def default_params(self):
        return {'in_steps': 30, 'in_cfg': 5, 'in_ckpt_model': 'a.safetensors',
                'in_ckpt_vae': None, 'in_vae_tiling': False,
                'in_upscl_tile_size': 128}

    def next_output_path(self, name='', extension='.png'):
        self.saved += 1                         # numbered like the app's own
        return str(self.tmp / f'{self.saved}{extension}')

    def generate_live(self, params, tab='img2img', keep=False):
        """An update of the app, as the tab shows it, then the images."""
        yield ("cmd", "progress", "stats", None)
        if tab == 'img2img':            # what the model draws
            yield ("cmd", progress_update(10, 5, 50), "stats", None)  # loading
            yield ("cmd", progress_update(40, 8, 20, "Speed: 2 it/s"),
                   "stats", [Image.new('RGB', (32, 32), (255, 0, 0))])
            yield ("cmd", progress_update(80, 16, 20, "Speed: 2 it/s"),
                   "stats", None)
        return self.generate(params, tab, keep)

    def generate(self, params, tab='img2img', keep=False):
        self.runs.append((tab, dict(params), keep))
        if self.stop_at and len(self.runs) == self.stop_at:
            raise GenerationStopped()
        if self.fail_at and len(self.runs) == self.fail_at:
            raise RuntimeError("no memory")
        if tab == 'txt2img':
            images = []
            for n in range(self.batch):
                image = Image.new('RGB', (100, 60), 'white')
                if keep:
                    image.filename = str(
                        self.tmp / ('base.png' if n == 0 else f'base{n}.png'))
                images.append(image)
            return images
        if tab == 'upscale':
            if not self.upscaler:
                raise ValueError("no generation for the tab 'upscale'")
            with Image.open(params['in_img_inp']) as small:
                return [small.resize((small.width * 4, small.height * 4))]
        with Image.open(params['in_img_inp']) as tile:
            assert tile.size == (params['in_width'], params['in_height'])
            return [Image.new('RGB', tile.size, (9, 9, 9))]


@pytest.fixture()
def api(tmp_path):
    return FakeApi(tmp_path)


def test_the_settings_start_as_the_apps_and_are_kept(api):
    cfg = settings.load(api)
    assert cfg['steps'] == 30 and cfg['cfg'] == 5.0
    assert cfg['checkpoint'] == 'a.safetensors'
    settings.save(api, {**cfg, 'denoise': 0.5, 'mode': tiles.CHESS,
                        'unknown': 1})
    again = settings.load(api)
    assert again['denoise'] == 0.5 and again['mode'] == tiles.CHESS
    assert 'unknown' not in again


def test_a_broken_settings_file_or_value_is_ignored(api, tmp_path):
    (tmp_path / settings.FILE).write_text('{"denoise": "lots", "steps": 12.0}')
    cfg = settings.load(api)
    assert cfg['denoise'] == 0.3 and cfg['steps'] == 12
    (tmp_path / settings.FILE).write_text('not json')
    assert settings.load(api)['denoise'] == 0.3


def test_a_tile_uses_the_prompt_and_model_of_the_image_unless_chosen(api):
    cfg = settings.defaults(api)
    image = {'in_pprompt': "a cat", 'in_nprompt': "ugly",
             'in_ckpt_model': 'image.safetensors', 'in_diffusion_mode': 0,
             'in_sampling': 'euler', 'in_scheduler': 'simple',
             'in_cnnet_bool': True, 'in_cnnet': 'pose', 'in_hires_bool': True}
    same = settings.tile_params(api, cfg, image, 5)
    assert same['in_ckpt_model'] == 'image.safetensors'
    assert (same['in_pprompt'], same['in_nprompt']) == ("a cat", "ugly")
    assert same['in_sampling'] == 'euler' and same['in_seed'] == 5
    # nothing of the image's own extras leaks into the tiles
    assert not same['in_cnnet_bool'] and not same['in_hires_bool']
    assert same['in_batch_count'] == 1 and same['in_img_mask'] is None
    cfg.update(model_from=settings.CUSTOM, checkpoint='mine.safetensors',
               prompt="a dog", negative="", sampler='dpm++2m')
    own = settings.tile_params(api, cfg, image, 5)
    assert own['in_ckpt_model'] == 'mine.safetensors'
    assert own['in_pprompt'] == "a dog" and own['in_nprompt'] == "ugly"
    assert own['in_sampling'] == 'dpm++2m'


def test_a_custom_unet_sets_its_own_files(api):
    cfg = {**settings.defaults(api), 'model_from': settings.CUSTOM,
           'model_type': settings.UNET, 'unet': 'u.gguf', 'llm': 'l.gguf'}
    params = settings.tile_params(api, cfg, {}, 1)
    assert params['in_diffusion_mode'] == 1 and params['in_unet_model'] == (
        'u.gguf')
    assert params['in_llm'] == 'l.gguf' and params['in_clip_l'] is None


def test_the_controlnet_and_tiled_decode_follow_the_settings(api):
    cfg = {**settings.defaults(api), 'controlnet': True,
           'cn_model': 'tile.safetensors', 'cn_strength': 0.8,
           'cn_union': True, 'cn_type': 'tile', 'tiled_decode': True}
    params = settings.tile_params(api, cfg, {}, 1)
    assert params['in_cnnet_bool'] and params['in_cnnet'] == (
        'tile.safetensors')
    assert params['in_control_strength'] == 0.8 and params['in_cnnet_union']
    assert params['in_control_type'] == 'tile' and params['in_vae_tiling']
    cfg['cn_model'] = ""                    # nothing chosen: no ControlNet
    assert not settings.tile_params(api, cfg, {}, 1)['in_cnnet_bool']


# -------------------------------------------------------------- extension

def _image(tmp_path, size=(100, 60)):
    return Image.new('RGB', size, 'white')


def _ctx(tab, **params):
    ctx = type('Ctx', (), {})()
    ctx.tab, ctx.params = tab, params
    return ctx


def test_it_takes_over_the_image_tabs_only_when_turned_on(api):
    on = {'ext_ultimate_sdcpp_upscale_enabled': True}
    assert extension.replace(api, _ctx('txt2img')) is None
    for tab in ('txt2img', 'img2img', 'imgedit'):
        assert extension.replace(api, _ctx(tab, **on)) is not None
    assert extension.replace(api, _ctx('any2video', **on)) is None


def test_the_pipeline_generates_then_upscales_each_image(api, tmp_path):
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128, 'seed': 0})
    progress, paths = run(extension.pipeline(
        api, 'txt2img', {'in_pprompt': "a cat"}))
    tab, params, keep = api.runs[0]
    assert (tab, keep) == ('txt2img', True)         # the tab's own image
    assert all(run[0] == 'img2img' and not run[2] for run in api.runs[1:])
    assert {run[1]['in_seed'] for run in api.runs[1:]} == {0}
    # the upscale and the image it comes from
    assert paths == [str(tmp_path / '1.png'), str(tmp_path / 'base.png')]
    with Image.open(paths[0]) as result:
        assert result.size == (200, 120)
        assert result.text['parameters'].startswith("a cat")
        assert "Ultimate SDCPP Upscale: x2.0" in result.text['parameters']
    assert progress[-1][0] == 100
    assert progress[0] == ("cmd", "progress", "stats", None)  # the tab's own
    ours = [p for p in progress if isinstance(p, Progress)]
    assert all(isinstance(p.image, Image.Image) for p in ours[1:]
               if p.image is not None)
    with Image.open(paths[0]) as saved:
        assert "Upscale original: base.png" in saved.text['parameters']
    assert not os.path.exists(tmp_path / 'tile.png')    # cleaned up


def test_the_upscale_model_is_used_then_the_image_is_resized(api, tmp_path):
    settings.save(api, {**settings.defaults(api), 'upscaler': 'esrgan.pth',
                        'mode': tiles.NO_TILES})
    _, paths = run(extension.pipeline(api, 'txt2img', {}))
    assert [r[0] for r in api.runs] == ['txt2img', 'upscale']
    assert api.runs[1][1]['in_upscl'] == 'esrgan.pth'
    with Image.open(paths[0]) as result:
        assert result.size == (200, 120)            # not the model's x4


def test_an_upscaler_that_can_not_run_is_an_error_not_a_resize(api):
    api.upscaler = False            # the app has no generation for it
    settings.save(api, {**settings.defaults(api), 'upscaler': 'esrgan.pth',
                        'mode': tiles.NO_TILES})
    _, paths = run(extension.pipeline(api, 'txt2img', {}))
    assert [os.path.basename(p) for p in paths] == ['base.png']   # original


def test_the_controlnet_gets_every_tile_as_its_control_image(api):
    settings.save(api, {**settings.defaults(api), 'controlnet': True,
                        'cn_model': 'tile.safetensors',
                        'tile_width': 128, 'tile_height': 128})
    run(extension.pipeline(api, 'txt2img', {}))
    for _, params, _ in api.runs[1:]:
        assert params['in_cnnet_bool'] and (
            params['in_control_img'] == params['in_img_inp'])


def _small_tiles(api):
    """Two tiles for the 200x120 upscale of the fake image."""
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128})


def test_stopping_saves_what_was_upscaled_and_keeps_the_original(
    api, tmp_path
):
    _small_tiles(api)
    api.stop_at = 3                         # while the second tile starts
    progress, paths = run(extension.pipeline(api, 'txt2img', {}))
    partial = tmp_path / '1_stopped.png'
    assert paths == [str(partial), str(tmp_path / 'base.png')]
    with Image.open(partial) as saved:
        assert saved.size == (200, 120)                 # the full size
        assert saved.getpixel((10, 10)) == (9, 9, 9)    # the first tile
        assert saved.getpixel((190, 110)) == (255, 255, 255)   # not yet
        assert "stopped after 1 of 2 areas" in saved.text['parameters']
        assert "Upscale original: base.png" in saved.text['parameters']
    assert "Stopped" in progress[-1].text
    assert "1_stopped.png" in progress[-1].text and "base.png" in (
        progress[-1].text)
    assert not os.path.exists(tmp_path / 'tile.png')
    assert not os.path.exists(tmp_path / '1.png')       # not passed as done


def test_stopping_before_the_first_tile_still_saves_the_resized_image(
    api, tmp_path
):
    _small_tiles(api)
    api.stop_at = 2
    _, paths = run(extension.pipeline(api, 'txt2img', {}))
    with Image.open(tmp_path / '1_stopped.png') as saved:
        assert saved.size == (200, 120)
        assert saved.getpixel((10, 10)) == (255, 255, 255)
        assert "stopped after 0 of 2 areas" in saved.text['parameters']
    assert paths[-1] == str(tmp_path / 'base.png')


def test_a_failure_also_saves_the_partial_and_keeps_the_original(
    api, tmp_path, capsys
):
    _small_tiles(api)
    api.fail_at = 3
    progress, paths = run(extension.pipeline(api, 'txt2img', {}))
    assert paths == [str(tmp_path / '1_partial.png'),
                     str(tmp_path / 'base.png')]
    with Image.open(paths[0]) as saved:
        assert "failed after 1 of 2 areas" in saved.text['parameters']
    assert "failed (no memory)" in progress[-1].text
    assert "RuntimeError: no memory" in capsys.readouterr().out


def test_the_images_finished_before_a_stop_are_kept_too(api, tmp_path):
    """With a batch: the ones done stay done, the one cut short is partial."""
    _small_tiles(api)
    api.batch = 2
    api.stop_at = 5            # txt2img, the 2 tiles of the 1st, 1 of the 2nd
    _, paths = run(extension.pipeline(api, 'txt2img', {}))
    assert paths == [str(tmp_path / '1.png'),          # done
                     str(tmp_path / '2_stopped.png'),  # cut short
                     str(tmp_path / 'base1.png')]      # its original


def test_each_tile_has_its_own_progress_bar_under_the_main_one(api):
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128})
    progress, _ = run(extension.pipeline(api, 'txt2img', {}))
    ours = [p for p in progress if isinstance(p, Progress)]
    assert ours[0].sub is None and "Upscaling" in ours[0].text
    bars = [p.sub for p in ours if p.sub]
    assert (0, "Tiles 1 · starting") in bars
    assert (10, "Tiles 1 · loading 5/50 · 10%") in bars   # then the steps
    assert (40, "Tiles 1 · step 8/20 · 40%") in bars
    assert (80, "Tiles 1 · step 16/20 · 80%") in bars
    assert ours[-1].sub is None and ours[-1].text == "Done"     # nothing left
    # the main bar counts the tiles and never goes back
    mains = [p.percent for p in ours]
    assert mains == sorted(mains) and mains[-1] == 100


def test_the_main_bar_is_there_from_the_first_moment_of_the_upscale(api):
    settings.save(api, {**settings.defaults(api), 'upscaler': 'esrgan.pth'})
    progress, _ = run(extension.pipeline(api, 'txt2img', {}))
    first = next(p for p in progress if isinstance(p, Progress))
    assert first.text == "Upscaling with esrgan.pth..."
    assert isinstance(first.image, Image.Image)    # and the image to see


def test_every_tile_shows_what_the_model_draws(api, tmp_path, monkeypatch):
    monkeypatch.setattr(extension, 'FRAME_EVERY', 0)
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128})
    progress, _ = run(extension.pipeline(api, 'txt2img', {}))
    frames = [p.image for p in progress
              if isinstance(p, Progress) and p.image is not None]
    reds = [f for f in frames if any(
        r > 200 and g < 90 and b < 90
        for _, (r, g, b) in f.convert('RGB').getcolors(maxcolors=10 ** 7))]
    assert len(reds) >= 2                       # the live tile is in the frame
    assert api.runs[1][1]['in_preview_bool']    # and was asked for


def test_the_frames_are_not_drawn_for_every_step(api, tmp_path):
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128})
    fast, _ = run(extension.pipeline(api, 'txt2img', {}))
    live = [p for p in fast if isinstance(p, Progress) and p.image]
    assert len(live) < 2 * 3 + 3                # throttled, not every step


def test_the_live_preview_can_be_turned_off_or_follows_the_tab(api):
    cfg = settings.defaults(api)
    quick = settings.tile_params(api, cfg, {}, 1)
    assert quick['in_preview_bool'] and quick['in_preview_mode'] == 'proj'
    own = settings.tile_params(api, cfg, {
        'in_preview_bool': True, 'in_preview_mode': 'tae',
        'in_preview_interval': 3}, 1)
    assert own['in_preview_mode'] == 'tae' and own['in_preview_interval'] == 3
    off = settings.tile_params(api, {**cfg, 'live_preview': False}, {}, 1)
    assert off['in_preview_bool'] is False


def test_the_saved_info_is_the_images_own_plus_the_upscale(api):
    cfg = settings.defaults(api)
    source = ('a cat\nNegative prompt: bad\nSteps: 35, Seed: 7, '
              'Size: 100x60, SDCPP: {"schema":"sdcpp.generation/v1"}')
    params = settings.tile_params(api, cfg, {}, 5)
    text = settings.parameters_text(
        cfg, params, (200, 120), 5, "3.png", "", source)
    head, _, record = text.rpartition(", SDCPP: ")
    assert record == '{"schema":"sdcpp.generation/v1"}'   # still the last
    assert head.startswith(source.split(", SDCPP: ")[0])
    assert "Ultimate SDCPP Upscale: x2.0" in head
    assert "Upscale size: 200x120" in head
    assert "Upscale original: 3.png" in head
    # An image without a record of its own just gets the fields added
    plain = settings.parameters_text(
        cfg, params, (200, 120), 5, "", "", "a cat\nSteps: 9")
    assert plain.startswith("a cat\nSteps: 9, Ultimate SDCPP Upscale")


def test_the_server_gets_the_controlnet_and_is_put_back(api):
    api.before = "before"
    settings.save(api, {**settings.defaults(api), 'tile_width': 128,
                        'tile_height': 128, 'seed': 0, 'controlnet': True,
                        'cn_model': "tile.safetensors"})
    progress, paths = run(extension.pipeline(api, 'txt2img', {}))
    assert api.served[0]['in_cnnet'] == "tile.safetensors"
    assert api.unserved == ["before"]
    assert paths and progress[-1][0] == 100


def test_the_server_is_put_back_when_the_upscale_is_stopped(api):
    api.before = "before"
    _small_tiles(api)
    api.stop_at = 3
    run(extension.pipeline(api, 'txt2img', {}))
    assert api.unserved == ["before"]


def test_the_terminal_says_which_tile_is_about_to_run(api, capsys):
    _small_tiles(api)
    run(extension.pipeline(api, 'txt2img', {}))
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith("[Ultimate SDCPP Upscale]")]
    assert len(lines) == 2
    assert "Tiles 1 of 2" in lines[0] and "steps run" in lines[0]

import importlib.util
import os

import pytest
from PIL import Image

from extensions import masker


@pytest.fixture()
def mask_dir(tmp_path, monkeypatch):
    d = tmp_path / 'masks'
    d.mkdir()
    monkeypatch.setattr(masker, 'MASK_DIR', str(d))
    return d


def test_create_mask_requires_image():
    assert masker.create_mask(None, 'x') == (
        None, None, 'Upload an image first.'
    )


def test_create_mask_without_rembg():
    # rembg is normally absent here; the failure must surface as a
    # note, never as an exception.
    if importlib.util.find_spec('rembg') is not None:
        pytest.skip('rembg installed; cannot exercise the missing path')
    img = Image.new('RGB', (10, 10))
    mask, overlay, note = masker.create_mask(img, 'x')
    assert mask is None
    assert overlay is None
    assert note.startswith('rembg failed:')


def test_sanitize_name():
    assert masker._sanitize_name('my cat/01') == 'my-cat-01'
    assert masker._sanitize_name('..hidden') == 'hidden'
    assert masker._sanitize_name('').startswith('mask_')
    assert masker._sanitize_name(None).startswith('mask_')


def test_list_masks_without_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(masker, 'MASK_DIR', str(tmp_path / 'nope'))
    assert masker.list_masks() == []


def test_list_masks(mask_dir):
    (mask_dir / 'b.png').write_bytes(b'x')
    (mask_dir / 'a.png').write_bytes(b'x')
    (mask_dir / 'readme.txt').write_bytes(b'x')
    assert masker.list_masks() == ['a.png', 'b.png']


def test_preview_mask_missing(mask_dir):
    assert masker.preview_mask('nope.png') is None
    assert masker.preview_mask('') is None


def test_preview_mask(mask_dir):
    Image.new('L', (4, 4), 128).save(mask_dir / 'm.png')
    pm = masker.preview_mask('m.png')
    assert pm is not None and pm.size == (4, 4)


def test_mask_path_traversal(mask_dir):
    assert masker._mask_path('') is None
    assert masker._mask_path(None) is None
    assert masker._mask_path('../evil.png') is None
    assert masker._mask_path('..') is None
    p = masker._mask_path('m.png')
    assert p is not None
    assert p.endswith(os.path.join('masks', 'm.png'))


def test_apply_mask_validation(mask_dir):
    assert masker.apply_mask(None, 'm.png') == (
        None, 'Upload an image first.'
    )
    assert masker.apply_mask(Image.new('RGB', (4, 4)), '') == (
        None, 'Pick a saved mask first.'
    )
    assert masker.apply_mask(Image.new('RGB', (4, 4)), 'nope.png') == (
        None, 'Mask not found: nope.png'
    )


def test_apply_mask(mask_dir):
    Image.new('L', (8, 8), 255).save(mask_dir / 'full.png')
    Image.new('L', (8, 8), 0).save(mask_dir / 'zero.png')
    img = Image.new('RGB', (16, 16), (1, 2, 3))
    out, note = masker.apply_mask(img, 'full.png')
    assert out is not None
    assert out.getchannel('A').getpixel((0, 0)) == 255
    assert 'full.png' in note
    out2, _ = masker.apply_mask(img, 'zero.png')
    assert out2 is not None
    assert out2.getchannel('A').getpixel((0, 0)) == 0

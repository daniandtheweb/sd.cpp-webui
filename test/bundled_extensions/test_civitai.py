"""The Civitai extension: links, searching, files, downloading."""

import hashlib
import json
import re
import shutil
import tempfile
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest

from extensions.civitai import client, extension
from modules.extensions import net
from modules.extensions.api import api_for
from modules.extensions.host import downloads_log, extensions
from modules.extensions.loader import load_extensions
from modules.extensions.ui import render_extensions_info

CIVITAI = Path(__file__).resolve().parents[2] / 'extensions' / 'civitai'
SHA = hashlib.sha256(b'weights').hexdigest()


def file(**kw):
    return {'id': 1, 'name': 'cool.safetensors', 'type': 'Model',
            'sizeKB': 2048.0, 'pickleScanResult': 'Success',
            'virusScanResult': 'Success', 'hashes': {'SHA256': SHA.upper()},
            'downloadUrl': 'https://civitai.com/api/download/models/9',
            **kw}


def version(**kw):
    return {'id': 9, 'name': 'v1', 'baseModel': 'SD 1.5',
            'trainedWords': ['cool style'], 'images': [],
            'files': [file()], **kw}


def model(**kw):
    return {'id': 5, 'name': 'Cool', 'type': 'LORA',
            'stats': {'downloadCount': 1234}, 'modelVersions': [version()],
            **kw}


# ------------------------------------------------------------------ links

@pytest.mark.parametrize('text, expected', [
    ('https://civitai.com/models/16014', (16014, None)),
    ('civitai.com/models/7?foo=1&modelVersionId=8', (7, 8)),
    ('  16014 ', (16014, None)),
    ('', None),
    ('https://civitai.com.evil.org/models/5', None),
])
def test_links_and_ids(text, expected):
    assert client.parse_link(text) == expected


def test_urls_are_built_safely():
    url = client.api_url('civitai.red', 'models', query='a&b=c d', limit=5,
                         types=None, sort='')
    assert url == 'https://civitai.red/api/v1/models?query=a%26b%3Dc+d&limit=5'
    with pytest.raises(ValueError):
        client.api_url('evil.org', 'models')


def test_the_key_header():
    assert client.headers(' k ') == {'Authorization': 'Bearer k'}
    assert client.headers('') == {} == client.headers('  ')


# -------------------------------------------------------------- the client

class FakeApi:
    def __init__(self, answers=None, boom=None):
        self.answers, self.boom, self.asked, self.downloads = (
            answers or {}, boom, [], [])
        self.settings = {'site': 'civitai.com', 'api_key': '',
                         'show_nsfw': False}
        self.notes = []
        self.folder = tempfile.mkdtemp(prefix='civitai-test-')

    def data_dir(self):
        return self.folder

    def fetch_json(self, url, headers=None):
        self.asked.append((url, headers))
        return self.answers[url.split('/api/v1/')[1].split('?')[0]]

    def get_setting(self, key):
        return self.settings[key]

    def download(self, url, filename, kind=None, headers=None, sha256=None,
                 progress=None):
        self.downloads.append(dict(url=url, filename=filename, kind=kind,
                                   headers=headers, sha256=sha256))
        if self.boom:
            raise self.boom
        if progress:
            progress(5, 10)
        return {'name': filename, 'size': 2 * 1024 * 1024, 'sha256': SHA,
                'host': 'x'}

    def notify(self, message, level='info'):
        self.notes.append(message)


def test_search_asks_the_chosen_site_with_the_key():
    api = FakeApi({'models': {'items': [model(), 'junk', 5]}})
    found, following = client.search(
        api, 'civitai.red', 'KEY', 'cool', 'LORA', 'Newest', False)
    assert [m['name'] for m in found] == ['Cool'] and following is None
    url, headers = api.asked[0]
    assert url.startswith('https://civitai.red/api/v1/models?query=cool')
    assert 'types=LORA' in url and 'sort=Newest' in url
    assert 'nsfw=false' in url
    assert headers == {'Authorization': 'Bearer KEY'}


def test_the_base_model_filter_is_sent_once_per_model():
    api = FakeApi({'models': {'items': []}})
    client.search(api, 'civitai.com', '', '', '', '', False,
                  ['SDXL 1.0', ' Anima ', '', 5, 'Flux.1 D'])
    url = api.asked[0][0]
    assert url.count('baseModels=') == 3
    assert 'baseModels=SDXL+1.0' in url and 'baseModels=Anima' in url
    assert 'baseModels=Flux.1+D' in url


def test_a_custom_base_model_is_encoded_not_injected():
    api = FakeApi({'models': {'items': []}})
    client.search(api, 'civitai.com', '', '', '', '', False,
                  ['x&nsfw=true&limit=9999'])
    url = api.asked[0][0]
    assert url.count('nsfw=') == 1 and url.count('limit=') == 1
    assert 'baseModels=x%26nsfw%3Dtrue%26limit%3D9999' in url


def test_the_next_page_is_followed_only_on_the_same_site():
    first = {'items': [model()], 'metadata': {
        'nextPage': 'https://civitai.com/api/v1/models?cursor=2'}}
    api = FakeApi({'models': first})
    items, following = client.search(
        api, 'civitai.com', 'K', 'x', '', '', False)
    assert following == 'https://civitai.com/api/v1/models?cursor=2'
    items, end = client.more(api, 'civitai.com', 'K', following)
    assert api.asked[-1] == (following, {'Authorization': 'Bearer K'})
    for bad in ('https://evil.org/api/v1/models?x=1',
                'https://civitai.red/api/v1/models?x=1',
                'https://civitai.com/api/v1/models/5', None, 5):
        with pytest.raises(ValueError):
            client.more(api, 'civitai.com', 'K', bad)


def test_a_model_link_keeps_the_linked_version():
    m = model(modelVersions=[version(id=1), version(id=2), version(id=3)])
    found = client.find(FakeApi({'models/5': m}), 'civitai.com', '', 5, 2)
    assert [v['id'] for v in found['modelVersions']] == [2]
    found = client.find(FakeApi({'models/5': m}), 'civitai.com', '', 5, None)
    assert len(found['modelVersions']) == 3


@pytest.mark.parametrize('kind, folder', [
    ('Checkpoint', 'checkpoint'),
    ('VAE', 'vae'),
    (None, None),
])
def test_kinds(kind, folder):
    assert client.kind_of({'type': kind}) == folder


def test_only_safe_model_files_are_offered():
    v = version(files=[
        file(name='ok.safetensors'),
        file(name='pruned.ckpt', type='Pruned Model'),
        file(name='flux-ae.sft', type='VAE'),
        file(name='data.zip', type='Training Data'),
        file(name='cfg.yaml', type='Config'),
        file(name='flow.json', type='Workflow'),
        file(name='archive.zip', type='Archive'),
        file(name='vision.safetensors', type='CLIPVision'),
        file(name='other.safetensors', type='Other'),
        file(name='te.sft', type='Text Encoder'),
        file(name='evil.safetensors', pickleScanResult='Danger'),
        file(name='virus.safetensors', virusScanResult='Danger'),
        file(name='script.exe', type='Model'),
        file(name='nolink.safetensors', downloadUrl=''),
        'junk',
    ])
    got = [f['name'] for f in client.downloadable(model(), v)]
    # model files of any type: the ones without a folder are placed by hand
    assert got == ['ok.safetensors', 'pruned.ckpt', 'flux-ae.sft',
                   'vision.safetensors', 'other.safetensors', 'te.sft']


@pytest.mark.parametrize('model_type, file_type, folder', [
    ('Checkpoint', 'Model', 'checkpoint'),
    ('Checkpoint', 'VAE', 'vae'),
    ('TextualInversion', 'Model', 'embedding'),
    ('Workflows', 'Model', None),
])
def test_each_file_goes_where_its_own_type_says(
    model_type, file_type, folder
):
    got = client.kind_of_file({'type': model_type}, {'type': file_type})
    assert got == folder


@pytest.mark.parametrize('name, expected', [
    ('ok name (1).safetensors', 'ok name (1).safetensors'),
    ('a/b\\c.safetensors', 'a_b_c.safetensors'),
    ('na\x00me.safetensors', 'na_me.safetensors'),
])
def test_local_names_are_plain(name, expected):
    got = client.local_name(name)
    assert got == expected
    assert net.safe_filename(got, model=True) == got


def test_sha256_must_be_a_real_hash():
    assert client.sha256_of(file()) == SHA
    assert client.sha256_of(file(hashes={'SHA256': 'abc'})) is None
    assert client.sha256_of(file(hashes=None)) is None


# --------------------------------------------------------------- handlers


def test_a_base_model_alone_is_a_search():
    api = FakeApi({'models': {'items': [model()]}})
    out = extension.search(api, '', 'All types', 'Newest', ['SDXL 1.0'])
    assert len(out[0]) == 1 and 'baseModels=SDXL+1.0' in api.asked[0][0]


def test_search_problems_are_messages():
    api = FakeApi({'models': {'items': []}})
    assert extension.search(api, 'zzz', '', '', [])[-4] == 'Nothing found.'

    class Down(FakeApi):
        def fetch_json(self, url, headers=None):
            raise urllib.error.HTTPError(url, 401, 'no', Message(), None)

    assert 'API key' in extension.search(Down(), 'x', '', '', [])[-4]


def _book_search(pages):
    """An api with a chain of pages (page n links to n + 1), and page 1."""
    answers = {}
    for n, names in enumerate(pages):
        nxt = ({'nextPage': f'https://civitai.com/api/v1/models?c={n + 2}'}
               if n + 1 < len(pages) else {})
        answers[n] = {'items': [model(name=x) for x in names],
                      'metadata': nxt}

    class Paged(FakeApi):
        def fetch_json(self, url, headers=None):
            self.asked.append((url, headers))
            cursor = url.split('c=')[1] if 'c=' in url else '1'
            return answers[int(cursor) - 1]

    api = Paged()
    return api, extension.search(api, 'x', 'All types', 'Newest', [])


def _names(grid):
    """The model names in the grid of cards."""
    import html as _html
    return [_html.unescape(n) for n in
            re.findall(r'civ-name">([^<]*)<', grid)]


def test_turning_a_page_clears_the_pick_and_keeps_the_old_book_intact():
    api, first = _book_search([['A1'], ['B1']])
    book = first[1]
    out = extension.turn(api, book, 1)
    assert out[2] is None and out[4]['choices'] == []
    assert book['page'] == 0 and len(book['pages']) == 1     # not mutated


def test_a_failed_page_keeps_the_current_one_with_a_message():
    api, first = _book_search([['A1'], ['B1']])

    def fail(url, headers=None):
        raise urllib.error.HTTPError(url, 429, 'slow', Message(), None)

    api.fetch_json = fail
    out = extension.turn(api, first[1], 1)
    assert out[1]['page'] == 0 and 'too many requests' in out[-4]
    assert out[-1]['interactive'] is True                    # can retry


def test_download_passes_what_the_app_needs():
    api = FakeApi()
    api.settings['api_key'] = 'KEY'
    text = extension.download(
        api, [model()], 0, 0, 0, progress=lambda *a, **k: None)
    (call,) = api.downloads
    assert call == dict(
        url='https://civitai.com/api/download/models/9',
        filename='cool.safetensors', kind='lora',
        headers={'Authorization': 'Bearer KEY'}, sha256=SHA)
    assert 'Downloaded **cool.safetensors**' in text
    assert 'cool style' in text and 'SHA256' in text
    assert 'refresh' in api.notes[0]


def test_download_needs_a_real_pick():
    api = FakeApi()
    assert 'Pick a model' in extension.download(api, [], 0, 0, 0)
    assert 'Pick a model' in extension.download(api, [model()], 0, 0, 7)
    assert 'Pick a model' in extension.download(api, [model()], None, 0, 0)
    assert api.downloads == []


@pytest.mark.parametrize('chosen', [
    '../../etc',
    'LORA',
    5,
])
def test_save_to_only_takes_the_folders_the_app_offers(chosen):
    api = FakeApi()
    v = version(files=[file(name='x.safetensors', type='Other')])
    m = model(type='Workflows', modelVersions=[v])
    text = extension.download(api, [m], 0, 0, 0, chosen)
    assert 'Choose where to save' in text and api.downloads == []


@pytest.mark.parametrize('boom, expected', [
    (FileExistsError('x'), 'already have cool.safetensors'),
])
def test_download_failures_are_messages(boom, expected):
    text = extension.download(FakeApi(boom=boom), [model()], 0, 0, 0)
    assert expected in text


def test_names_from_civitai_are_escaped_in_the_page():
    evil = model(name='<script>alert(1)</script>',
                 modelVersions=[version(name='<img onerror=x>')])
    api = FakeApi()
    html_text = extension._info(api, evil, evil['modelVersions'][0])
    assert '<script>' not in html_text and '<img onerror' not in html_text
    assert '&lt;script&gt;' in html_text


def _images(*urls, level=1, kind='image'):
    return {'images': [{'url': u, 'nsfwLevel': level, 'type': kind}
                       for u in urls]}


GOOD = 'https://image.civitai.com/key/uuid/original=true/1.jpeg'


# ------------------------------------------------- with the real app parts

PUBLIC = '93.184.216.34'


@pytest.fixture()
def civitai(tmp_path, monkeypatch, config):
    """The real extension, loaded; folders in tmp; a fake internet."""
    ext_dir = tmp_path / 'exts'
    ext_dir.mkdir()
    shutil.copytree(CIVITAI, ext_dir / 'civitai',
                    ignore=shutil.ignore_patterns('__pycache__'))
    monkeypatch.setattr(config, 'config_path', str(tmp_path / 'cfg.json'))
    models = tmp_path / 'models' / 'loras'
    monkeypatch.setitem(config.data, 'lora_dir', str(models))
    monkeypatch.setattr(net, '_resolve', lambda host: [PUBLIC])
    assert load_extensions(
        SimpleNamespace(data={'ext_dir': str(ext_dir)})) == ['civitai']
    monkeypatch.setattr('modules.extensions.ui.ensure_loaded', lambda: None)
    return api_for(extensions()[0]), models, config


class Page:
    def __init__(self, data):
        import io
        self.body, self.headers = io.BytesIO(data), {
            'Content-Length': str(len(data))}

    def read(self, n=-1):
        return self.body.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def redirect(location):
    headers = Message()
    headers['Location'] = location
    return urllib.error.HTTPError('u', 307, 'r', headers, None)


@pytest.mark.parametrize('site', ['civitai.com', 'civitai.red'])
def test_a_real_download_from_either_site(civitai, monkeypatch, site):
    api, models, config = civitai
    monkeypatch.setitem(config.data, 'ext_civitai_site', site)
    monkeypatch.setitem(config.data, 'ext_civitai_api_key', 'SECRETKEY')
    start = f'https://{site}/api/download/models/9'
    storage = 'https://civitai-delivery.abc.r2.cloudflarestorage.com/f?sig=1'
    seen = []

    def urlopen(request):
        seen.append((request.full_url, dict(request.header_items())))
        if request.full_url == start:
            raise redirect(storage)
        assert request.full_url == storage
        return Page(b'weights')

    monkeypatch.setattr(net, '_urlopen', urlopen)
    m = model(modelVersions=[version(files=[file(downloadUrl=start)])])
    text = extension.download(api, [m], 0, 0, 0)
    assert 'Downloaded **cool.safetensors**' in text
    assert (models / 'cool.safetensors').read_bytes() == b'weights'
    first, second = seen
    assert first[1]['Authorization'] == 'Bearer SECRETKEY'
    assert 'Authorization' not in second[1]            # not to the storage
    (record,) = downloads_log()
    assert record['status'] == 'ok' and record['kind'] == 'lora'
    assert record['host'].endswith('r2.cloudflarestorage.com')
    assert 'SECRETKEY' not in str(record) and 'sig=1' not in str(record)


def test_a_login_page_instead_of_a_model_is_refused(civitai, monkeypatch):
    api, models, config = civitai
    monkeypatch.setattr(net, '_urlopen',
                        lambda request: Page(b'<html>log in</html>'))
    text = extension.download(api, [model()], 0, 0, 0)
    assert 'Not downloaded' in text and 'sha256' in text
    assert list(models.iterdir()) == []                # nothing left behind
    assert downloads_log()[0]['status'] == 'failed'


def test_a_file_from_another_host_is_refused(civitai, monkeypatch):
    api, models, config = civitai
    called = []
    monkeypatch.setattr(net, '_urlopen', lambda r: called.append(r))
    m = model(modelVersions=[version(
        files=[file(downloadUrl='https://evil.org/x.safetensors')])])
    assert 'not one of the extension' in extension.download(api, [m], 0, 0, 0)
    assert called == []


def test_an_existing_model_is_never_replaced(civitai, monkeypatch):
    api, models, config = civitai
    models.mkdir(parents=True)
    (models / 'cool.safetensors').write_bytes(b'mine')
    monkeypatch.setattr(net, '_urlopen', lambda r: Page(b'weights'))
    assert 'already have' in extension.download(api, [model()], 0, 0, 0)
    assert (models / 'cool.safetensors').read_bytes() == b'mine'


def test_the_key_is_a_secret_setting_with_a_password_field(civitai, config):
    api, _, _ = civitai
    (ext,) = extensions()
    assert ext.settings['api_key'].secret
    assert ext.settings['site'].choices == ('civitai.com', 'civitai.red')
    with gr.Blocks() as demo, gr.Tabs():
        render_extensions_info()
    boxes = [b for b in demo.blocks.values()
             if getattr(b, 'label', None) == 'API key']
    assert len(boxes) == 1
    assert boxes[0].type == 'password' and boxes[0].value == ''


def _save_secret(api, config, typed, clear=False):
    """Press Save in the settings of the extension with these values."""
    (ext,) = extensions()
    with gr.Blocks() as demo, gr.Tabs():
        render_extensions_info()
    save = next(f for f in demo.fns.values()
                if f.fn.__name__ == 'save_settings')
    count = len(save.inputs)
    values = {'Site': 'civitai.com', 'API key': typed,
              'Show NSFW models and previews': False,
              'Clear API key': clear}
    given = [values[c.label] for c in save.inputs]
    assert count == 4
    return save.fn(*given)


def test_saving_with_the_key_field_empty_keeps_the_key(civitai):
    api, _, config = civitai
    config.data['ext_civitai_api_key'] = 'KEEP'
    assert _save_secret(api, config, '') == 'Saved.'
    assert config.data['ext_civitai_api_key'] == 'KEEP'
    assert _save_secret(api, config, 'NEW') == 'Saved.'
    assert config.data['ext_civitai_api_key'] == 'NEW'


def test_the_key_can_be_cleared(civitai):
    api, _, config = civitai
    config.data['ext_civitai_api_key'] = 'KEEP'
    _save_secret(api, config, '', clear=True)
    assert config.data['ext_civitai_api_key'] == ''


# ----------------------------------------------------------- the gallery

GOOD = 'https://image.civitai.com/key/uuid/original=true/1.jpeg'


def _images(*urls, level=1, kind='image'):
    return {'images': [{'url': u, 'nsfwLevel': level, 'type': kind}
                       for u in urls]}


@pytest.mark.parametrize('url', [
    'https://evil.org/a.jpg',
    'javascript:alert(1)',
    'ftp://image.civitai.com/a.jpg',
])
def test_pictures_only_from_civitai(url):
    assert extension._pictures(_images(url), True, 'width=512', 8) == []


def test_nsfw_pictures_need_the_setting():
    v = _images(GOOD, level=4)
    assert extension._pictures(v, False, 'width=512', 8) == []
    assert len(extension._pictures(v, True, 'width=512', 8)) == 1


def test_next_and_previous_page_replace_the_grid():
    api, first = _book_search([['A1', 'A2'], ['B1'], ['C1', 'C2']])
    assert _names(first[3]) == ['A1', 'A2']

    out = extension.turn(api, first[1], 1)                  # next: fetched
    models, book, picked, grid, *_ = out
    assert _names(grid) == ['B1'] and book['page'] == 1
    assert [m['name'] for m in models] == ['B1']
    assert 'Page 2: 1 result' in out[-4]
    assert out[-3]['interactive'] is True                   # can go back
    assert out[-1]['interactive'] is True                   # more to come

    out = extension.turn(api, book, 1)
    assert _names(out[3]) == ['C1', 'C2'] and out[-1]['interactive'] is False
    book = out[1]

    asked = len(api.asked)
    out = extension.turn(api, book, -1)                     # back: cached
    assert _names(out[3]) == ['B1'] and len(api.asked) == asked
    out = extension.turn(api, out[1], -1)
    assert _names(out[3]) == ['A1', 'A2'] and out[-3]['interactive'] is False
    out = extension.turn(api, out[1], 1)                    # forward: cached
    assert _names(out[3]) == ['B1'] and len(api.asked) == asked


@pytest.mark.parametrize('bad', [
    None,
    True,
])
def test_a_wrong_click_changes_nothing(bad):
    index, grid, versions, files, info, previews = extension.pick(
        FakeApi(), [model()], bad)
    assert index is None and info == '' and previews['value'] == []


# ------------------------------------------------- base models, learned

def test_searching_teaches_the_filter_and_it_never_gets_in_the_way(tmp_path):
    api = FakeApi({'models': {'items': [model(
        modelVersions=[version(baseModel='Qwen 3.0')])]}})
    out = extension.search(api, 'x', 'All types', 'Newest', [])
    assert len(out[0]) == 1
    assert 'Qwen 3.0' in client.base_choices(
        extension._bases_path(api))

    class NoFolder(FakeApi):
        def data_dir(self):
            raise OSError('read-only disk')

    api = NoFolder({'models': {'items': [model()]}})
    assert len(extension.search(api, 'x', 'All types', 'Newest', [])[0]) == 1


# ------------------------------------------------------- the tab's wiring

def _tab(api):
    with gr.Blocks() as demo:
        extension.render_tab(api)
    return demo, list(demo.blocks.values())


# ------------------------------------------------- the model cards (HTML)

def test_card_text_is_escaped():
    m = model(name='<b onmouseover=x>Evil</b>',
              modelVersions=[version(baseModel='<i>base</i>')])
    out = extension.cards(FakeApi(), [m])
    assert '<b onmouseover' not in out and '<i>base' not in out
    assert '&lt;b onmouseover' in out


def test_search_fills_everything_for_a_link():
    api = FakeApi({'models/5': model(
        modelVersions=[version(**_images(GOOD))])})
    (models, book, picked, grid, versions, files, info, previews, status,
     previous, label, following) = extension.search(
        api, 'https://civitai.com/models/5', 'All types', 'Newest', [])
    assert len(models) == 1 and picked == 0
    assert book == {'pages': [models], 'nexts': [None], 'page': 0}
    assert 'civ-picked' in grid and _names(grid) == ['Cool']
    assert versions['choices'] == [('v1 (SD 1.5)', 0)]
    assert files['choices'][0][0].startswith('cool.safetensors')
    assert 'cool style' in info and 'files go to: lora' in info
    assert len(previews['value']) == 1
    assert 'Page 1: 1 result' in status and label == 1
    assert previous['interactive'] is False
    assert following['interactive'] is False


def test_search_by_text_shows_the_grid_and_waits_for_a_click():
    page = {'items': [model(), model(name='Other')],
            'metadata': {'nextPage': 'https://civitai.com/api/v1/models?c=2'}}
    api = FakeApi({'models': page})
    (models, book, picked, grid, versions, files, info, previews, status,
     previous, label, following) = extension.search(
        api, 'cool', 'All types', 'Newest', ['Anima'])
    assert picked is None and 'civ-picked' not in grid
    assert _names(grid) == ['Cool', 'Other']
    assert versions['choices'] == [] and info == ''
    assert previews['value'] == []
    assert book['nexts'] == ['https://civitai.com/api/v1/models?c=2']
    assert previous['interactive'] is False
    assert following['interactive'] is True
    assert 'baseModels=Anima' in api.asked[0][0]


# ----------------------------------------------------------- the layout

def _tab(api):
    with gr.Blocks() as demo:
        extension.render_tab(api)
    return demo, list(demo.blocks.values())


def _find(blocks, label):
    return next(b for b in blocks if getattr(b, 'label', None) == label)


def test_every_handler_returns_what_its_outputs_expect():
    """Gradio only complains about a wrong count when someone clicks."""
    api = FakeApi({'models': {'items': [model(
        modelVersions=[version(**_images(GOOD))])], 'metadata': {
        'nextPage': 'https://civitai.com/api/v1/models?c=2'}}})
    demo, blocks = _tab(api)

    def handler(label):
        widget = next(b for b in blocks if getattr(b, 'value', None) == label)
        return next(f for f in demo.fns.values()
                    if (widget._id, 'click') in f.targets)

    search = handler('Search')
    out = search.fn('cool', 'All types', 'Newest', [])
    assert len(out) == len(search.outputs) == 12

    book = out[1]
    for label in ('Next page', 'Previous page'):
        turn = handler(label)
        got = turn.fn(book)
        assert len(got) == len(turn.outputs) == 12
        book = got[1]

    grid = next(b for b in blocks if isinstance(b, gr.HTML)
                and 'civ-card' in (b.js_on_load or ''))
    card = next(f for f in demo.fns.values()
                if (grid._id, 'click') in f.targets)
    assert card.fn.__name__ == 'on_card'
    got = card.fn([model()], SimpleNamespace(index=0))
    assert len(got) == len(card.outputs) == 6

    version_box = _find(blocks, 'Version')
    change = next(f for f in demo.fns.values()
                  if (version_box._id, 'input') in f.targets)
    got = change.fn([model()], 0, 0)
    assert len(got) == len(change.outputs) == 3


def test_the_tab_builds_and_has_its_hosts_listed(civitai):
    demo, blocks = _tab(civitai[0])
    labels = {getattr(b, 'label', None) for b in blocks}
    assert {'Search or paste a link', 'Type', 'Base model', 'Sort',
            'Version', 'File', 'Previews'} <= labels
    (ext,) = extensions()
    assert 'civitai.red' in ext.hosts and '*.civitai.red' in ext.hosts


# ------------------------------------------- how the pictures were made

A1111_META = {
    'ENSD': '31337', 'Size': '512x768', 'seed': 1541450048, 'steps': 20,
    'prompt': 'masterpiece, 1girl, <lora:animeoutlineV4_16:1>',
    'sampler': 'DPM++ 2M Karras', 'cfgScale': 7, 'Clip skip': '2',
    'negativePrompt': 'EasyNegative, badhandv4',
    'resources': [{'name': 'animeoutlineV4_16', 'type': 'lora', 'weight': 1},
                  {'name': 'Dream', 'type': 'model', 'hash': 'abc'}],
    'Model hash': '6e430eb514', 'Hires upscaler': 'SwinIR_4x',
    'Hires upscale': '2.5', 'Denoising strength': '0.5',
}

COMFY_GRAPH = {'prompt': {
    '3': {'class_type': 'KSampler', 'inputs': {
        'seed': 453349536654733, 'steps': 9, 'cfg': 1.0,
        'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0,
        'positive': ['6', 0], 'negative': ['7', 0]}},
    '6': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'a red fox'}},
    '7': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'blurry'}},
}}


def test_a_pictures_meta_is_read_into_plain_text():
    info = client.describe(A1111_META)
    assert info['prompt'].startswith('masterpiece, 1girl')
    assert info['negative'] == 'EasyNegative, badhandv4'
    settings = dict(info['settings'])
    assert settings['Steps'] == '20' and settings['Seed'] == '1541450048'
    assert settings['Sampler'] == 'DPM++ 2M Karras'
    assert settings['CFG scale'] == '7' and settings['Clip skip'] == '2'
    assert settings['Size'] == '512x768'
    assert settings['Hires upscaler'] == 'SwinIR_4x'
    assert [label for label, _ in info['settings']][:3] == [
        'Steps', 'Sampler', 'CFG scale']                    # fixed order
    assert info['resources'] == ['lora: animeoutlineV4_16 (1)',
                                 'model: Dream']


def test_a_comfyui_picture_gives_its_prompts_and_sampler():
    meta = {'comfy': json.dumps(COMFY_GRAPH), 'seed': 5, 'vaes': ['ae.sft']}
    info = client.describe(meta)
    assert (info['prompt'], info['negative']) == ('a red fox', 'blurry')
    settings = dict(info['settings'])
    assert settings['Steps'] == '9' and settings['Sampler'] == 'euler'
    assert settings['Scheduler'] == 'simple' and settings['CFG scale'] == '1.0'
    assert settings['Seed'] == '5'                       # the meta's own wins
    assert settings['VAE'] == 'ae.sft'


@pytest.mark.parametrize('comfy', [
    None,
    'x' * (client.COMFY_LIMIT + 1),
])
def test_an_unreadable_workflow_is_just_no_comfy_data(comfy):
    assert client.comfy({'comfy': comfy}) in ({}, {'prompt': '',
                                                    'negativePrompt': ''})
    client.describe({'comfy': comfy, 'prompt': 'p'})          # never raises


@pytest.mark.parametrize('meta', [
    None,
    {'junk': 1},
])
def test_pictures_without_data_describe_as_empty(meta):
    info = client.describe(meta)
    assert (info['prompt'], info['negative']) == ('', '')
    assert info['resources'] == [] or meta == {'junk': 1}


def test_what_the_author_wrote_is_bounded_and_text_only():
    info = client.describe({
        'prompt': 'p' * 50000, 'steps': {'x': 1}, 'sampler': ['a'],
        'Size': 'q' * 1000, 'resources': ['junk', {'name': ''}, 5,
                                           {'name': 'ok', 'weight': {}}]})
    assert len(info['prompt']) == client.TEXT_LIMIT
    assert dict(info['settings']).get('Size') == 'q' * client.VALUE_LIMIT
    assert 'Steps' not in dict(info['settings'])          # not a value
    assert 'Sampler' not in dict(info['settings'])
    assert info['resources'] == ['resource: ok']


# ---- clicking a picture

def _picked(meta=A1111_META):
    """Models with one version whose pictures aaa and bbb are shown."""
    urls = [f'https://image.civitai.com/k/{n}/original=true/{n}.jpg'
            for n in ('aaa', 'bbb')]
    m = model(modelVersions=[version(id=9, **_images(*urls))])
    api = FakeApi({'model-versions/9': {'images': [
        {'url': urls[0], 'meta': meta},
        {'url': urls[1], 'meta': None}]}})
    return api, [m]


def test_clicking_a_picture_shows_how_it_was_made():
    api, models = _picked()
    prompt, negative, table, metas = extension.show_details(
        api, models, 0, 0, {}, 0)
    assert prompt.startswith('masterpiece') and negative.startswith('Easy')
    assert '>Steps</td><td>20</td>' in table
    assert 'lora: animeoutlineV4_16 (1)' in table
    assert list(metas) == ['civitai.com:9']


def test_the_meta_of_a_version_is_asked_once():
    api, models = _picked()
    _, _, _, metas = extension.show_details(api, models, 0, 0, {}, 0)
    extension.show_details(api, models, 0, 0, metas, 1)
    extension.show_details(api, models, 0, 0, metas, 0)
    assert len(api.asked) == 1


def test_what_the_author_wrote_is_escaped_in_the_table():
    api, models = _picked({'steps': '<script>alert(1)</script>',
                           'sampler': '<img src=x onerror=y>',
                           'resources': [{'name': '<b>x</b>'}]})
    _, _, table, _ = extension.show_details(api, models, 0, 0, {}, 0)
    assert '<script>' not in table and '<img' not in table
    assert '<b>x' not in table and '&lt;script&gt;' in table


@pytest.mark.parametrize('index', [
    None,
    True,
])
def test_a_bad_click_shows_the_hint_and_asks_nothing(index):
    api, models = _picked()
    assert extension.show_details(api, models, 0, 0, {}, index) == (
        '', '', extension.HINT, {})
    assert api.asked == []


def test_a_failed_request_is_a_message_and_is_tried_again_later():
    api, models = _picked()

    def fail(url, headers=None):
        raise urllib.error.HTTPError(url, 429, 'slow', Message(), None)

    real = api.fetch_json
    api.fetch_json = fail
    prompt, negative, table, metas = extension.show_details(
        api, models, 0, 0, {}, 0)
    assert 'too many requests' in table and metas == {}
    api.fetch_json = real
    assert extension.show_details(api, models, 0, 0, metas, 0)[0]


def test_the_details_follow_the_picture_that_nsfw_hides():
    """The gallery hides NSFW pictures: its index must match ours."""
    urls = ['https://image.civitai.com/k/aaa/original=true/1.jpg',
            'https://image.civitai.com/k/bbb/original=true/2.jpg']
    m = model(modelVersions=[version(id=9, images=[
        {'url': urls[0], 'nsfwLevel': 4, 'type': 'image'},      # hidden
        {'url': urls[1], 'nsfwLevel': 1, 'type': 'image'}])])
    api = FakeApi({'model-versions/9': {'images': [
        {'url': urls[0], 'meta': {'prompt': 'hidden one'}},
        {'url': urls[1], 'meta': {'prompt': 'shown one'}}]}})
    assert extension.show_details(api, [m], 0, 0, {}, 0)[0] == 'shown one'


# ---------------------------------------------------------- safe by default

def test_nsfw_is_off_until_the_user_turns_it_on(civitai):
    api, _, config = civitai
    (ext,) = extensions()
    assert ext.settings['show_nsfw'].default is False
    assert api.get_setting('show_nsfw') is False


@pytest.mark.parametrize('image, safe', [
    ({'nsfwLevel': 1}, True),
    ({'nsfwLevel': 4}, False),
    ({}, False),
    ({'nsfwLevel': 'x'}, False),
    ({'nsfw': 'Mature'}, False),
])
def test_a_picture_is_safe_only_when_civitai_rates_it_so(image, safe):
    assert extension._is_safe(image) is safe


def test_an_unrated_picture_needs_the_setting():
    v = {'images': [{'url': GOOD, 'type': 'image'}]}   # no nsfwLevel at all
    assert extension._pictures(v, False, 'width=512', 8) == []
    assert len(extension._pictures(v, True, 'width=512', 8)) == 1


@pytest.mark.parametrize('flagged, expected', [
    ({'nsfw': True}, True),
    ({}, False),
    # nsfwLevel adds up the levels of everything a model shows: safe models
    # that Civitai returns for nsfw=false have values like these
    ({'nsfw': False, 'nsfwLevel': 7}, False),
    ({'nsfw': False, 'nsfwLevel': 25}, False),
    ({'nsfw': False, 'nsfwLevel': 31}, False),
])
def test_only_the_models_own_flag_makes_it_nsfw(flagged, expected):
    assert client.is_nsfw(model(**flagged)) is expected


def test_nsfw_models_in_a_search_are_dropped_unless_the_setting_is_on():
    answer = {'items': [model(name='Safe'),
                        model(name='Spicy', nsfw=True)]}
    api = FakeApi({'models': answer})
    out = extension.search(api, 'x', 'All types', 'Newest', [])
    assert [m['name'] for m in out[0]] == ['Safe']
    assert 'nsfw=false' in api.asked[0][0]
    api = FakeApi({'models': answer})
    api.settings['show_nsfw'] = True
    out = extension.search(api, 'x', 'All types', 'Newest', [])
    assert [m['name'] for m in out[0]] == ['Safe', 'Spicy']
    assert 'nsfw=true' in api.asked[0][0]


def test_a_linked_nsfw_model_says_why_it_is_not_shown():
    api = FakeApi({'models/5': model(nsfw=True)})
    out = extension.search(api, 'https://civitai.com/models/5', '', '', [])
    assert out[0] == [] and 'marked NSFW' in out[-4]
    api.settings['show_nsfw'] = True
    out = extension.search(api, 'https://civitai.com/models/5', '', '', [])
    assert len(out[0]) == 1


def test_a_search_keeps_every_model_the_server_calls_safe():
    """The first page must not shrink: nsfwLevel is a mix, not a rating."""
    levels = [25, 7, 3, 3, 7, 11, 31, 15, 1]
    answer = {'items': [model(id=n, nsfw=False, nsfwLevel=level)
                        for n, level in enumerate(levels)]}
    out = extension.search(FakeApi({'models': answer}), 'x', 'All types',
                           'Newest', [])
    assert len(out[0]) == len(levels)


def test_a_page_asks_for_ten_models_at_most():
    api = FakeApi({'models': {'items': [model()]}})
    extension.search(api, 'x', 'All types', 'Newest', [])
    assert 'limit=10' in api.asked[0][0]


def test_the_tab_says_which_mode_is_on(civitai):
    api, _, config = civitai
    assert 'SFW</span> Only safe for work' in extension._mode(api)
    config.data['ext_civitai_show_nsfw'] = True
    badge = extension._mode(api)
    assert 'NSFW</span> NSFW models and pictures are shown' in badge
    assert 'Extensions tab' in badge

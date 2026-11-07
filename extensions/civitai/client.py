"""Civitai's API, and what to do with what it answers.

civitai.com and civitai.red share one API. Nothing here touches the
network itself: it asks the app's api.fetch_json (see extension.py), which
only reaches the hosts declared in HOSTS.
"""

import json
import re
from urllib.parse import urlencode, urlsplit

SITES = ('civitai.com', 'civitai.red')

# What Civitai calls a model -> the kind the app knows (api.download)
KINDS = {
    'Checkpoint': 'checkpoint',
    'LORA': 'lora', 'LoCon': 'lora', 'DoRA': 'lora',
    'TextualInversion': 'embedding',
    'VAE': 'vae',
    'Controlnet': 'controlnet',
    'Upscaler': 'upscaler',
}
SEARCH_TYPES = ['Checkpoint', 'LORA', 'TextualInversion', 'Controlnet',
                'VAE', 'Upscaler']
# Models per page: the page is a grid of cards with a picture each, and the
# browser has to draw all of them
PAGE_SIZE = 10

SORTS = ["Highest Rated", "Most Downloaded", 'Newest']
# Base models to filter by, as Civitai spells them: the ones seen on its
# front page when this was written, most used first. Civitai adds new ones
# all the time ("Qwen 2.1", "MOSS Image"...), so the extension also learns
# every base model it sees in results (see remember) and offers those too.
BASE_MODELS = [
    "SD 1.5", 'Illustrious', 'Pony', "SDXL 1.0", 'Anima', "Flux.1 D",
    "Krea 2", 'NoobAI', 'ZImageTurbo', "MOSS Image", 'Other', "ACE Audio",
    "Krea 2 Turbo", "SD 1.4", "MiniMax H3", 'Qwen', "SDXL Lightning",
    'ZImageBase', "Flux.2 Klein 9B", 'Chroma', "SD 1.5 LCM",
    "Wan Video 2.2 I2V-A14B", "Qwen 2.1", "Qwen 2", "Flux.1 S",
    "Hunyuan Video", "Wan Video 2.2 T2V-A14B", "Flux.2 Klein 9B-base",
    "SDXL Hyper", "Wan Video 14B t2v", "SD 2.1 768", "SDXL 1.0 LCM",
    "Wan Video", "Flux.2 Klein 4B", "Wan Video 14B i2v 720p",
    "Flux.2 Klein 4B-base", "Flux.3 Klein", "LTXV 2.5",
    "Wan Video 2.2 TI2V-5B", "Wan Video 1.3B t2v", "SD 2.0", 'Ernie',
    "SDXL 0.9", "Flux.1 Krea", "Wan Video 14B i2v 480p", "LTXV 2.3",
    "Flux.2 D", "Pony V7", 'Kling', 'Boogu', "SD 1.5 Hyper",
    "Ideogram 4.0", "Flux.3 Klein-base", 'OpenAI', 'HiDream', 'Flux.3',
    'Lumina', 'Upscaler', 'LTXV2', "SD 2.1",
]
LEARNED_FILE = 'base_models.json'
MAX_LEARNED = 300

# Where the app can put a file (the kinds of api.download), by what to call
# them in the page. "auto" means: by the file's own type.
AUTO = 'auto'
DESTINATIONS = [
    ("Automatic (by file type)", AUTO),
    ('Checkpoints', 'checkpoint'),
    ("Diffusion models (unet)", 'diffusion'),
    ('VAE', 'vae'),
    ("Text encoders", 'text_encoder'),
    ('LoRAs', 'lora'),
    ('Embeddings', 'embedding'),
    ('ControlNets', 'controlnet'),
    ('Upscalers', 'upscaler'),
    ('ADetailer', 'adetailer'),
    ('IP-Adapters', 'ipadapter'),
    ('PuLID', 'pulid'),
    ('AnimateDiff', 'animatediff'),
    ('PhotoMaker', 'photomaker'),
    ('TAESD', 'taesd'),
]

# Files of a model version that have their own folder, whatever the type
# of the model: a checkpoint's page can offer its VAE and text encoder
FILE_KINDS = {
    'VAE': 'vae',
    "Diffusion Model": 'diffusion', 'UNet': 'diffusion',
    "Text Encoder": 'text_encoder',
}
# Files that are the model itself: they go where the model's type says
MODEL_FILE_TYPES = ('Model', "Pruned Model")
# Training data, workflows, configs and archives are not offered: they are
# not model files (zip, json, yaml, txt). Any other type of model file is,
# for the user to place.
MODEL_FORMATS = ('.safetensors', '.sft', '.gguf', '.ckpt', '.pt', '.pth',
                 '.bin')

_LINK = re.compile(
    r'(?:https?://)?(?:www\.)?civitai\.(?:com|red)/'
    r'(?:models/(\d+)(?:[^#\s]*?[?&]modelVersionId=(\d+))?'
    r'|api/download/models/(\d+))', re.I)


def parse_link(text: str) -> tuple[int | None, int | None] | None:
    """
    (model id, version id) of a Civitai link or a bare model id; None when
    the text is something to search for.
    """
    text = (text or '').strip()
    if text.isdigit():
        return int(text), None
    found = _LINK.search(text)
    if not found:
        return None
    model, version, download = found.groups()
    if download:
        return None, int(download)
    return int(model), int(version) if version else None


def headers(key: str) -> dict:
    return {'Authorization': f"Bearer {key.strip()}"} if key.strip() else {}


def api_url(site: str, path: str, **params) -> str:
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}")
    query = urlencode(
        {k: v for k, v in params.items() if v not in (None, '', [], ())},
        doseq=True,
    )
    return f'https://{site}/api/v1/{path}' + (f'?{query}' if query else '')


def _items(data, key=None) -> list:
    data = data.get(key) if key and isinstance(data, dict) else data
    return [i for i in data if isinstance(i, dict)] if isinstance(
        data, list) else []


def _clean_bases(bases) -> list[str]:
    return [b.strip() for b in (bases or []) if isinstance(b, str)
            and b.strip()][:10]


def _page(answer) -> tuple[list[dict], str | None]:
    """(models, url of the next page) of an answer of the models list."""
    if not isinstance(answer, dict):
        return [], None
    meta = answer.get('metadata')
    following = meta.get('nextPage') if isinstance(meta, dict) else None
    return _items(answer, 'items'), following if isinstance(
        following, str) else None


def search(
    api, site, key, text, kind, sort, nsfw, bases=(), limit=PAGE_SIZE
) -> tuple[list[dict], str | None]:
    """
    The models for a search text, as Civitai describes them, and the url
    of the next page (None at the end). bases filters by base model.
    """
    url = api_url(
        site, 'models', query=text.strip(),
        types=kind if kind in SEARCH_TYPES else None,
        sort=sort if sort in SORTS else None, limit=limit,
        baseModels=_clean_bases(bases),
        nsfw='true' if nsfw else 'false',
    )
    return _page(api.fetch_json(url, headers(key)))


def more(api, site, key, following) -> tuple[list[dict], str | None]:
    """The next page of a search: only urls of the models list of the site."""
    if not (isinstance(following, str)
            and following.startswith(f'https://{site}/api/v1/models?')):
        raise ValueError("not the next page of a search")
    return _page(api.fetch_json(following, headers(key)))


def find(api, site, key, model_id, version_id) -> dict:
    """One model, from a link: it has its versions (or the one linked)."""
    if model_id is not None:
        model = api.fetch_json(
            api_url(site, f'models/{model_id}'), headers(key))
        if not isinstance(model, dict):
            raise ValueError("unexpected answer")
        model = dict(model)
        if version_id is not None:
            model['modelVersions'] = [
                v for v in _items(model, 'modelVersions')
                if v.get('id') == version_id
            ] or model.get('modelVersions', [])
        return model
    version = api.fetch_json(
        api_url(site, f'model-versions/{version_id}'), headers(key))
    if not isinstance(version, dict):
        raise ValueError("unexpected answer")
    info = version.get('model') or {}
    return {'id': version.get('modelId'), 'name': info.get('name', ''),
            'type': info.get('type', ''), 'modelVersions': [version]}


# ------------------------------------------------------------------ files

def is_nsfw(model: dict) -> bool:
    """
    Whether the model carries Civitai's own NSFW flag. Its nsfwLevel is
    not used: it adds up the levels of everything the model shows, so
    most safe models have bits above Mature set too.
    """
    return bool(model.get('nsfw'))


def kind_of(model: dict) -> str | None:
    return KINDS.get(str(model.get('type')))


def kind_of_file(model: dict, file: dict) -> str | None:
    """Where a file goes: by its own type, else by the model's type."""
    kind = FILE_KINDS.get(str(file.get('type')))
    if kind:
        return kind
    if file.get('type') in MODEL_FILE_TYPES:
        return kind_of(model)
    return None


def local_name(name: str) -> str:
    """A file name the app accepts, as close to Civitai's as possible."""
    name = re.sub(r"[^\w.()+,@=\- ]", '_', str(name)).strip(" .")
    return name[:200]


def downloadable(model: dict, version: dict) -> list[dict]:
    """The model files of a version that weren't flagged by the scans."""
    files = []
    for f in _items(version, 'files'):
        name = str(f.get('name') or '')
        flagged = 'Danger' in (f.get('pickleScanResult'),
                               f.get('virusScanResult'))
        if (not flagged and name.lower().endswith(MODEL_FORMATS)
                and f.get('downloadUrl')):
            files.append(f)
    return files


def destination(model: dict, file: dict, chosen: str | None) -> str | None:
    """
    The kind to save a file as: the one the user chose, else the one its
    type says (None when it has none: the user has to choose).
    """
    if chosen and chosen != AUTO:
        return chosen if chosen in dict(
            (value, label) for label, value in DESTINATIONS) else None
    return kind_of_file(model, file)


def size_text(kilobytes) -> str:
    if not isinstance(kilobytes, (int, float)):
        return '?'
    mb = kilobytes / 1024
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB" if mb >= 1 else f"{kilobytes:.0f} KB"


def sha256_of(file: dict) -> str | None:
    value = (file.get('hashes') or {}).get('SHA256')
    return value.lower() if isinstance(value, str) and re.fullmatch(
        r'[0-9a-fA-F]{64}', value) else None


# ------------------------------------------------------- base models seen

def _base_names(models) -> set[str]:
    """The base models of the versions in some models, as plain text."""
    names = set()
    for model in models:
        for version in _items(model, 'modelVersions'):
            name = version.get('baseModel')
            if (isinstance(name, str) and 0 < len(name.strip()) <= 60
                    and name.isprintable()):
                names.add(name.strip())
    return names


def _learned(path: str) -> set[str]:
    try:
        with open(path, encoding='utf-8') as f:
            names = json.load(f)
    except (OSError, ValueError):
        return set()
    return {n for n in names if isinstance(n, str)} if isinstance(
        names, list) else set()


def remember(path: str, models) -> None:
    """Keep the base models seen in results, so the filter can offer them."""
    known = _learned(path)
    names = known | _base_names(models)
    if names == known or len(names) > MAX_LEARNED:
        return
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sorted(names), f)
    except OSError:
        pass


def base_choices(path: str) -> list[str]:
    """The filter's choices: the starting list, then what was learned."""
    extra = sorted(_learned(path) - set(BASE_MODELS), key=str.lower)
    return BASE_MODELS + extra


# ------------------------------------------------ how a picture was made

TEXT_LIMIT = 6000
VALUE_LIMIT = 200
COMFY_LIMIT = 2_000_000   # bytes of workflow we are willing to read

# (label, the keys Civitai's meta may use for it), in the order shown
SETTINGS = [
    ('Steps', ('steps',)),
    ('Sampler', ('sampler', 'sampler_name')),
    ('Scheduler', ('scheduler', "Schedule type")),
    ("CFG scale", ('cfgScale', 'cfg', "CFG scale")),
    ('Seed', ('seed',)),
    ('Size', ('Size', 'size')),
    ("Clip skip", ('clipSkip', "Clip skip")),
    ("Denoising strength", ("Denoising strength", 'denoise')),
    ('Model', ('Model', 'model')),
    ("Model hash", ("Model hash",)),
    ('VAE', ('VAE', 'vae')),
    ("Hires upscaler", ("Hires upscaler",)),
    ("Hires upscale", ("Hires upscale",)),
    ("Hires steps", ("Hires steps",)),
    ("Hires sampler", ("Hires sampler",)),
]


def image_key(url) -> str:
    """
    What identifies a picture in all of Civitai's urls: its id in the path
    (.../<id>/original=true/1.jpeg and .../<id>/width=512/1.jpeg match).
    """
    parts = [p for p in urlsplit(str(url)).path.split('/') if p]
    return parts[-3] if len(parts) >= 3 else ''


def version_meta(api, site, key, version_id) -> dict[str, dict]:
    """
    How the pictures of a version were made: {picture id: meta}, only for
    those that have it. (Civitai's search doesn't carry it; this asks the
    version, which lists up to 10 pictures.)
    """
    if not isinstance(version_id, int) or isinstance(version_id, bool):
        raise ValueError("not a version")
    answer = api.fetch_json(
        api_url(site, f'model-versions/{version_id}'), headers(key))
    found = {}
    for image in _items(answer, 'images'):
        meta = image.get('meta')
        picture = image_key(image.get('url'))
        if picture and isinstance(meta, dict) and meta:
            found[picture] = meta
    return found


def _text(value, limit=VALUE_LIMIT) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (str, int, float)):
        return str(value).strip()[:limit]
    return ''


def _follow(graph: dict, link, depth=0) -> str:
    """The text a ComfyUI link ends in: through the conditioning nodes."""
    if depth > 8 or not (isinstance(link, list) and link):
        return ''
    node = graph.get(str(link[0]))
    inputs = node.get('inputs') if isinstance(node, dict) else None
    if not isinstance(inputs, dict):
        return ''
    if isinstance(inputs.get('text'), str):
        return inputs['text']
    for name, value in inputs.items():
        if name.startswith('conditioning') and isinstance(value, list):
            found = _follow(graph, value, depth + 1)
            if found:
                return found
    return ''


def comfy(meta: dict) -> dict:
    """
    What can be read of a ComfyUI picture's workflow: the positive and
    negative prompts, and the first sampler's settings. {} if it isn't
    readable; this only reads, it never runs anything.
    """
    raw = meta.get('comfy')
    if not isinstance(raw, str) or len(raw) > COMFY_LIMIT:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        return {}
    graph = data.get('prompt') if isinstance(data, dict) else None
    if not isinstance(graph, dict):
        return {}
    for node in graph.values():
        if not (isinstance(node, dict)
                and 'KSampler' in str(node.get('class_type'))):
            continue
        inputs = node.get('inputs')
        if not isinstance(inputs, dict):
            continue
        found = {
            'prompt': _follow(graph, inputs.get('positive')),
            'negativePrompt': _follow(graph, inputs.get('negative')),
        }
        for name in ('steps', 'cfg', 'sampler_name', 'scheduler', 'seed',
                     'denoise'):
            if isinstance(inputs.get(name), (str, int, float)):
                found[name] = inputs[name]
        return found
    return {}


def _resources(meta: dict) -> list[str]:
    """'lora: name (0.8)' for each resource listed, as plain text."""
    lines = []
    resources = meta.get('resources')
    for item in resources if isinstance(resources, list) else []:
        if not isinstance(item, dict) or not _text(item.get('name')):
            continue
        weight = item.get('weight')
        lines.append(
            f"{_text(item.get('type')) or 'resource'}: "
            f"{_text(item.get('name'))}"
            + (f" ({_text(weight)})" if _text(weight) else '')
        )
    return lines[:30]


def describe(meta) -> dict:
    """
    How a picture was made, as plain text for display: {'prompt',
    'negative', 'settings': [(label, value)], 'resources': [text]}. ComfyUI
    pictures fill in what the usual fields don't have.
    """
    if not isinstance(meta, dict) or not meta:
        return {'prompt': '', 'negative': '', 'settings': [],
                'resources': []}
    merged = {**comfy(meta), **{k: v for k, v in meta.items()
                                if k != 'comfy' and v not in (None, '')}}
    settings = []
    for label, keys in SETTINGS:
        for key in keys:
            value = _text(merged.get(key))
            if value:
                settings.append((label, value))
                break
    vaes = meta.get('vaes')
    if isinstance(vaes, list) and all(label != 'VAE' for label, _ in settings):
        names = ", ".join(_text(v) for v in vaes if _text(v))
        if names:
            settings.append(('VAE', names[:VALUE_LIMIT]))
    return {
        'prompt': _text(merged.get('prompt'), TEXT_LIMIT),
        'negative': _text(merged.get('negativePrompt'), TEXT_LIMIT),
        'settings': settings,
        'resources': _resources(meta),
    }

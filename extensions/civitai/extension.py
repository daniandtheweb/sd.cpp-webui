"""Civitai: search and download models from civitai.com or civitai.red.

Search by name, or paste a model or version link. Pick a version and a
file and press Download: the app saves it in the model folder of its kind
(checkpoints, LoRAs, embeddings, VAEs, ControlNets, upscalers) and lists
it in the Downloads log of the Extensions tab. The extension never sees
the model folders. Some files need an API key (Civitai > Account settings
> API Keys): set it in the Extensions tab, in the Civitai settings.
"""

import html
import os
import urllib.error
from urllib.parse import urlsplit

import gradio as gr

from . import client

NAME = 'Civitai'
VERSION = '1.0'
AUTHOR = 'daniandtheweb'
URL = 'https://github.com/daniandtheweb/sd.cpp-webui'
DESCRIPTION = (
    "Search Civitai (civitai.com or civitai.red) and download models "
    "straight into the right model folder."
)
API_VERSION = 1
# The sites, and where they send the files (their storage)
HOSTS = {
    'civitai.com': "the model search and details (its API)",
    '*.civitai.com': "the previews and the model files",
    'civitai.red': "the same, on the civitai.red mirror",
    '*.civitai.red': "the previews and the model files of the mirror",
    '*.r2.cloudflarestorage.com': "where Civitai stores the model files",
    '*.backblazeb2.com': "where Civitai stores some model files",
}

ALL_TYPES = "All types"


def register(api):
    api.add_setting(
        'site', client.SITES[0], label='Site', choices=client.SITES,
        info="civitai.com or its civitai.red mirror.")
    api.add_setting(
        'api_key', '', label="API key", secret=True,
        info="Needed by some files. Kept in the config file as text; "
             "sent to the site you chose only.")
    api.add_setting(
        'show_nsfw', False, label="Show NSFW models and previews")
    api.add_tab('Civitai', lambda: render_tab(api))


# --------------------------------------------------------------- handlers

def _settings(api):
    return (api.get_setting('site'), api.get_setting('api_key'),
            bool(api.get_setting('show_nsfw')))


def _problem(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
        return ("Civitai wants you logged in for this: set your API key "
                "in the Extensions tab (Civitai settings).")
    if isinstance(e, urllib.error.HTTPError) and e.code == 429:
        return "Civitai says too many requests: wait a moment."
    return f"{type(e).__name__}: {e}"


def _choices(items, label):
    return [(label(item), n) for n, item in enumerate(items)]


def _version_label(version: dict) -> str:
    return f"{version.get('name', '?')} ({version.get('baseModel') or '?'})"


def _file_label(model: dict, file: dict) -> str:
    """Name, size and where it will be saved: 'x.safetensors (diffusion)'."""
    kind = client.kind_of_file(model, file) or 'choose where in "Save to"'
    return (f"{file.get('name')}  -  "
            f"{client.size_text(file.get('sizeKB'))}  -  {kind}")


def _versions(model):
    return [v for v in model.get('modelVersions', []) if isinstance(v, dict)]


COVER = 'width=256'      # cards: small, the grid has many
PREVIEW = 'width=512'    # previews: big enough to look at


def _is_safe(image: dict) -> bool:
    """
    Whether Civitai rates a picture as safe (level 1, "None"). One that
    isn't rated is not safe: it only shows with the NSFW setting on.
    """
    level = image.get('nsfwLevel')
    if isinstance(level, int) and not isinstance(level, bool):
        return level <= 1
    return image.get('nsfw') == 'None'


def _picture(image: dict, nsfw: bool, size: str) -> str | None:
    """
    The url of a picture the browser may load, resized by Civitai's image
    server (a few hundred KB, not 3 MB), or None: only Civitai's own hosts,
    only images, and the ones that are not rated safe only when the NSFW
    setting is on (it is off until the user turns it on).
    """
    full = str(image.get('url'))
    parts = urlsplit(full)
    host = parts.hostname or ''
    if (image.get('type') != 'image' or parts.scheme != 'https'
            or not host.endswith(('.civitai.com', '.civitai.red'))
            or not (nsfw or _is_safe(image))):
        return None
    return full.replace('/original=true/', f'/{size}/')


def _pictures(version: dict, nsfw: bool, size: str,
              limit: int | None = None) -> list[str]:
    """The pictures of a version we may show (all, unless limit says)."""
    urls = (_picture(i, nsfw, size) for i in version.get('images') or []
            if isinstance(i, dict))
    return [url for url in urls if url][:limit]


def _cover(model: dict, nsfw: bool) -> str:
    """The first picture of the model we may show, small."""
    for version in _versions(model):
        found = _pictures(version, nsfw, COVER, 1)
        if found:
            return found[0]
    return ''


def _card(n: int, model: dict, selected: bool, nsfw: bool) -> str:
    versions = _versions(model)
    base = versions[0].get('baseModel') if versions else None
    count = (model.get('stats') or {}).get('downloadCount')
    meta = " · ".join(str(part) for part in (
        model.get('type'), base,
        f"{count:,} downloads" if isinstance(count, int) else None,
    ) if part)
    cover = _cover(model, nsfw)
    picture = (
        f'<img src="{html.escape(cover, quote=True)}" loading="lazy" '
        'referrerpolicy="no-referrer">' if cover else ''
    )
    return (
        f'<div class="civ-card{" civ-picked" if selected else ""}" '
        f'data-index="{n}"><div class="civ-cover">{picture}</div>'
        f'<div class="civ-name">{html.escape(str(model.get("name")))}</div>'
        f'<div class="civ-meta">{html.escape(meta)}</div></div>'
    )


def cards(api, models: list, selected: int | None = None) -> str:
    """The grid of models, as the page shows it (names are escaped)."""
    nsfw = _settings(api)[2]
    return ('<div class="civ-grid">' + ''.join(
        _card(n, model, n == selected, nsfw)
        for n, model in enumerate(models)
    ) + '</div>')


def _no_previews():
    """An empty square: the gallery stays, so the layout doesn't jump."""
    return gr.update(value=[])


def _previews(api, version: dict):
    _, _, nsfw = _settings(api)
    urls = _pictures(version, nsfw, PREVIEW)
    return gr.update(value=urls)


def _learn(api, models: list):
    """Remember the base models seen, for the filter. Never in the way."""
    try:
        client.remember(_bases_path(api), models)
    except Exception:
        pass


def _bases_path(api) -> str:
    return os.path.join(api.data_dir(), client.LEARNED_FILE)


def _empty():
    return gr.update(choices=[], value=None)


def _nav(book: dict) -> tuple:
    """
    The paging controls: the previous button (enabled when there is a page
    before), the page number, and the next button.
    """
    page = book['page']
    has_next = bool(book['nexts'][page]) or page + 1 < len(book['pages'])
    return (gr.update(interactive=page > 0), page + 1,
            gr.update(interactive=has_next))


def _mode(api) -> str:
    """The badge that says which mode is on: SFW (the default) or NSFW."""
    if _settings(api)[2]:
        label, color = "NSFW", "#c0392b"
        text = "NSFW models and pictures are shown."
    else:
        label, color = "SFW", "#1e8e3e"
        text = "Only safe for work models and pictures are shown."
    return (
        f'<span style="background:{color};color:#fff;padding:3px 10px;'
        f'border-radius:6px;font-weight:600">{label}</span> {text} '
        'Change it with "Show NSFW models and previews" in the '
        'Civitai settings of the Extensions tab.'
    )


def _nothing(status: str) -> tuple:
    """Every output of the search emptied, with a message."""
    book = {'pages': [[]], 'nexts': [None], 'page': 0}
    return ([], book, None, '', _empty(), _empty(), '', _no_previews(),
            status, *_nav(book))


def search(api, text, kind, sort, bases):
    """
    Search, or open a link; with nothing to search for it browses what
    Civitai lists first (by the sort and filters). Returns (models, book,
    picked, grid, versions, files, info, previews, status, previous, page,
    next), in the order of the tab's outputs. The book keeps the pages
    fetched, so going back doesn't ask again.
    """
    site, key, nsfw = _settings(api)
    link = client.parse_link(text)
    following = None
    try:
        if link:
            models = [client.find(api, site, key, *link)]
        else:
            models, following = client.search(
                api, site, key, text or '', kind, sort, nsfw, bases)
    except Exception as e:
        return _nothing(_problem(e))
    # Civitai is asked for safe models only, but the answer is checked too
    hidden = 0
    if not nsfw:
        shown = [m for m in models if not client.is_nsfw(m)]
        hidden, models = len(models) - len(shown), shown
    if not models:
        return _nothing(
            "This model is marked NSFW: turn on \"Show NSFW models and "
            "previews\" in the Extensions tab to see it." if link and hidden
            else "Nothing found.")

    _learn(api, models)
    book = {'pages': [models], 'nexts': [following], 'page': 0}
    picked = 0 if link else None
    parts = (_empty(), _empty(), '', _no_previews())
    if link:
        parts = show_model(api, models, 0)
    return (models, book, picked, cards(api, models, picked), *parts,
            _page_text(book, site), *_nav(book))


def _page_text(book: dict, site: str) -> str:
    models = book['pages'][book['page']]
    return (f"Page {book['page'] + 1}: {len(models)} result(s) from "
            f"{site}. Click one.")


def turn(api, book, step):
    """
    Go to the next (step 1) or previous (step -1) page: (models, book,
    picked, grid, versions, files, info, previews, status, previous, page,
    next). A page
    not seen yet is fetched; if that fails the page stays and the status
    says why.
    """
    site, key, _ = _settings(api)
    page = book['page'] + step
    if not 0 <= page < len(book['pages']) + (1 if step > 0 else 0):
        return _stay(api, book, site, None)
    if page == len(book['pages']) and not book['nexts'][book['page']]:
        return _stay(api, book, site, None)
    book = {k: list(v) if isinstance(v, list) else v
            for k, v in book.items()}
    if page == len(book['pages']):
        try:
            models, following = client.more(
                api, site, key, book['nexts'][book['page']])
        except Exception as e:
            return _stay(api, book, site, _problem(e))
        if not models:
            book['nexts'][book['page']] = None
            return _stay(api, book, site, "That was the last page.")
        _learn(api, models)
        book['pages'].append(models)
        book['nexts'].append(following)
    book['page'] = page
    models = book['pages'][page]
    return (models, book, None, cards(api, models), _empty(), _empty(), '',
            _no_previews(), _page_text(book, site), *_nav(book))


def _stay(api, book, site, problem):
    """The same page again, with a message (or the page's own)."""
    models = book['pages'][book['page']]
    return (models, book, *[gr.skip()] * 6,
            problem or _page_text(book, site), *_nav(book))


def pick(api, models, index):
    """A model was clicked: (picked, grid, versions, files, info, previews)."""
    if (not isinstance(index, int) or isinstance(index, bool)
            or not 0 <= index < len(models)):
        return None, gr.update(), _empty(), _empty(), '', _no_previews()
    return (index, cards(api, models, index),
            *show_model(api, models, index))


def _info(api, model, version) -> str:
    site = _settings(api)[0]
    files = client.downloadable(model, version)
    kinds = sorted({k for k in (client.kind_of_file(model, f) for f in files)
                    if k})
    unplaced = any(not client.kind_of_file(model, f) for f in files)
    where = ", ".join(
        part for part in (
            f"files go to: {', '.join(kinds)}" if kinds else '',
            'some files need you to choose where in "Save to"'
            if unplaced else '',
        ) if part
    ) or "no model files to download here"
    words = ", ".join(
        str(w) for w in (version.get('trainedWords') or []))
    link = (f"https://{site}/models/{model.get('id')}"
            f"?modelVersionId={version.get('id')}")
    lines = [
        f"<b>{html.escape(str(model.get('name')))}</b> - "
        f"{html.escape(_version_label(version))}",
        f'{where}',
        f"Trigger words: <code>{html.escape(words)}</code>" if words
        else '',
        f'<a href="{html.escape(link, quote=True)}" target="_blank" '
        f'rel="noopener noreferrer">open on {html.escape(site)}</a>',
    ]
    return '<div>' + '<br>'.join(line for line in lines if line) + '</div>'


HINT = "Click a picture to see how it was made."
NO_DATA = ("No generation data for this picture: the author didn't share "
           "it, or it is beyond the first 10 pictures of the version.")


def _row(label: str, value: str) -> str:
    return (f'<tr><td style="padding:2px 14px 2px 0;opacity:.7">'
            f'{html.escape(label)}</td><td>{html.escape(value)}</td></tr>')


def _details_html(info: dict) -> str:
    """The settings and resources of a picture, as a table (escaped)."""
    parts = []
    if info['settings']:
        parts.append('<table>' + ''.join(
            _row(label, value) for label, value in info['settings'])
            + '</table>')
    if info['resources']:
        parts.append('<div style="margin-top:8px"><b>Resources</b><br>'
                     + '<br>'.join(html.escape(r) for r in info['resources'])
                     + '</div>')
    return ''.join(parts)


def show_details(api, models, picked, version_index, metas, index):
    """
    A preview was clicked: (prompt, negative prompt, settings, metas). The
    pictures' meta comes from the version, asked once and kept in metas.
    """
    site, key, nsfw = _settings(api)
    try:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise IndexError
        version = _versions(models[picked])[version_index]
        url = _pictures(version, nsfw, PREVIEW)[index]
    except (TypeError, IndexError):
        return '', '', HINT, metas
    metas = dict(metas or {})
    where = f"{site}:{version.get('id')}"
    if where not in metas:
        try:
            metas[where] = client.version_meta(
                api, site, key, version.get('id'))
        except Exception as e:
            return '', '', html.escape(_problem(e)), metas
    info = client.describe(metas[where].get(client.image_key(url)))
    table = _details_html(info)
    if not (info['prompt'] or info['negative'] or table):
        return '', '', NO_DATA, metas
    return info['prompt'], info['negative'], table, metas


def clear_details():
    return '', '', HINT


def show_model(api, models, index):
    """A model was picked: (versions, files, info, previews)."""
    empty = gr.update(choices=[], value=None)
    try:
        versions = _versions(models[index])
    except (TypeError, IndexError):
        return empty, empty, '', _no_previews()
    if not versions:
        return empty, empty, "This model has no versions.", _no_previews()
    update = gr.update(
        choices=_choices(versions, _version_label), value=0)
    return (update, *show_version(api, models, index, 0))


def show_version(api, models, index, version_index):
    """A version was picked: (files, info, previews)."""
    try:
        model = models[index]
        version = _versions(model)[version_index]
    except (TypeError, IndexError):
        return gr.update(choices=[], value=None), '', _no_previews()
    files = client.downloadable(model, version)
    update = gr.update(
        choices=_choices(files, lambda f: _file_label(model, f)),
        value=0 if files else None)
    return (update, _info(api, model, version), _previews(api, version))


def download(api, models, index, version_index, file_index,
             target=client.AUTO, progress=gr.Progress()):
    """
    Download the picked file into the folder of its kind (or the one chosen
    in target); returns the message to show.
    """
    try:
        model = models[index]
        version = _versions(model)[version_index]
        file = client.downloadable(model, version)[file_index]
    except (TypeError, IndexError):
        return "Pick a model, a version and a file first."
    kind = client.destination(model, file, target)
    if kind is None:
        return ('Choose where to save this file in "Save to": its type '
                "doesn't say.")

    _, key, _ = _settings(api)
    name = client.local_name(file['name'])

    def report(done, total):
        if total:
            progress((done, total), unit='bytes')

    try:
        result = api.download(
            file['downloadUrl'], name, kind=kind,
            headers=client.headers(key), sha256=client.sha256_of(file),
            progress=report)
    except FileExistsError:
        return f"You already have {name}: nothing downloaded."
    except Exception as e:
        return f"Not downloaded: {_problem(e)}"
    words = ", ".join(str(w) for w in (version.get('trainedWords') or []))
    api.notify(f"Downloaded {name}. Press the refresh button next to the "
               "model list to use it.")
    return (f"Downloaded **{html.escape(name)}** as a {kind} "
            f"({client.size_text(result['size'] / 1024)}), verified"
            f"{' with its SHA256' if client.sha256_of(file) else ''}."
            + (f"\n\nTrigger words: `{words}`" if words else ''))


# --------------------------------------------------------------------- UI

GRID_CSS = """
.civ-grid { display: grid; gap: 10px; max-height: 560px; overflow-y: auto;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            padding: 4px; }
.civ-card { cursor: pointer; border-radius: 8px; padding: 6px;
            border: 2px solid transparent;
            background: var(--background-fill-secondary); }
.civ-card:hover { border-color: var(--border-color-primary); }
.civ-card.civ-picked { border-color: var(--color-accent); }
.civ-cover { aspect-ratio: 2 / 3; border-radius: 6px; overflow: hidden;
             background: var(--background-fill-primary); }
.civ-cover img { width: 100%; height: 100%; object-fit: cover; }
.civ-name { font-weight: 600; margin-top: 6px; font-size: 0.9em;
            overflow: hidden; display: -webkit-box;
            -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.civ-meta { font-size: 0.75em; opacity: 0.7; margin-top: 2px; }
"""

# A click anywhere in the grid says which card it was
GRID_JS = """
element.addEventListener('click', (event) => {
  const card = event.target.closest('.civ-card');
  if (card) trigger('click', {index: Number(card.dataset.index)});
});
"""

def render_tab(api):
    gr.Markdown(
        "## Civitai\n\nSearch by name, or paste a model or version link "
        "or id. Click a model to see its versions and pictures. Downloads "
        "go to the model folder of their kind and show up in the "
        "Downloads log of the Extensions tab."
    )
    mode = gr.HTML(_mode(api))
    models = gr.State([])
    book = gr.State({'pages': [[]], 'nexts': [None], 'page': 0})
    picked = gr.State(None)
    metas = gr.State({})
    with gr.Row():
        text = gr.Textbox(
            label="Search or paste a link", scale=4,
            placeholder="a name, https://civitai.com/models/123, 123 ...")
        go = gr.Button('Search', variant='primary', scale=0)
    with gr.Row():
        kind = gr.Dropdown(
            [ALL_TYPES] + client.SEARCH_TYPES, value=ALL_TYPES,
            label='Type', scale=1)
        bases = gr.Dropdown(
            client.base_choices(_bases_path(api)), value=[], multiselect=True,
            allow_custom_value=True, label="Base model", scale=2,
            info="SDXL 1.0, Anima, Qwen... or type another. New ones show "
                 "up as they appear in results.")
        sort = gr.Dropdown(
            client.SORTS, value=client.SORTS[0], label='Sort', scale=1)
    # The paging controls sit above the results, like the Gallery's
    with gr.Group():
        with gr.Row(equal_height=True):
            previous = gr.Button("Previous page", interactive=False)
            page_number = gr.Number(
                label="Page", value=1, interactive=False, precision=0)
            following = gr.Button("Next page", interactive=False)
    grid = gr.HTML(
        '', html_template='${value}', css_template=GRID_CSS,
        js_on_load=GRID_JS)
    # What the search did ("Page 1: 10 result(s)...") is read under the grid
    status = gr.Markdown()

    # The chosen model: its download settings on the left, its pictures
    # on the right
    with gr.Row():
        with gr.Column():
            version = gr.Dropdown(label='Version', choices=[])
            file = gr.Dropdown(label='File', choices=[])
            info = gr.HTML()
            target = gr.Dropdown(
                client.DESTINATIONS, value=client.AUTO, label="Save to",
                info="Where the file goes. Automatic follows its type.")
            pull = gr.Button('Download', variant='primary')
            downloaded = gr.Markdown()
        with gr.Column():
            # Three columns of Gradio's square items; as many rows as the
            # version has pictures
            previews = gr.Gallery(
                label='Previews', columns=3, object_fit='cover',
                interactive=False, buttons=[])
            prompt = gr.Textbox(
                label='Prompt', lines=4, max_lines=10, interactive=False,
                buttons=['copy'])
            negative = gr.Textbox(
                label="Negative prompt", lines=2, max_lines=8,
                interactive=False, buttons=['copy'])
            settings = gr.HTML(HINT)

    # The outputs of a search, and what a click on a card changes
    found = [models, book, picked, grid, version, file, info, previews,
             status, previous, page_number, following]
    inputs = [text, kind, sort, bases]

    def show_mode():
        return _mode(api)

    # The mode is read again with every search: the setting may have changed
    go.click(lambda *a: search(api, *a), inputs=inputs,
             outputs=found).then(show_mode, outputs=[mode])
    # The first page of models is there when the page opens
    gr.on(fn=lambda: search(api, '', ALL_TYPES, client.SORTS[0], []),
          outputs=found).then(show_mode, outputs=[mode])
    text.submit(lambda *a: search(api, *a), inputs=inputs,
                outputs=found).then(show_mode, outputs=[mode])
    previous.click(lambda b: turn(api, b, -1), inputs=[book],
                   outputs=found).then(show_mode, outputs=[mode])
    following.click(lambda b: turn(api, b, 1), inputs=[book],
                    outputs=found).then(show_mode, outputs=[mode])

    def on_card(models, evt: gr.EventData):
        return pick(api, models, getattr(evt, 'index', None))

    grid.click(on_card, inputs=[models],
               outputs=[picked, grid, version, file, info, previews])
    version.input(
        lambda m, i, v: show_version(api, m, i, v),
        inputs=[models, picked, version], outputs=[file, info, previews])
    # A click on a picture shows how it was made; a new set clears that
    def on_preview(models, picked, version, metas, evt: gr.SelectData):
        return show_details(api, models, picked, version, metas, evt.index)

    previews.select(
        on_preview, inputs=[models, picked, version, metas],
        outputs=[prompt, negative, settings, metas])
    previews.change(clear_details, outputs=[prompt, negative, settings])
    # The filter also offers the base models seen since the page opened
    bases.focus(lambda: gr.update(
        choices=client.base_choices(_bases_path(api))), outputs=[bases])

    def pull_file(m, i, v, f, t, progress=gr.Progress()):
        return download(api, m, i, v, f, t, progress)

    pull.click(pull_file, inputs=[models, picked, version, file, target],
               outputs=[downloaded])

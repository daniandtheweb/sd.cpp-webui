# Extensions

Extensions add features without touching the app. The app itself stays
minimal: an extension brings the packages it needs and the app installs
them, once you agree.

An extension is either

- a **single file**: `extensions/my_ext.py`, or
- a **folder**: `extensions/my_ext/extension.py`, next to its own helper
  modules (`from . import helper` works) and data.

Extensions are **opt-in**: start the app with `--extensions`. Without the
flag nothing is discovered, installed or imported, and no Extensions tab
is shown.

The folder is the config key `ext_dir` (when unset, the app's own
`extensions/` folder, resolved relative to the app and created if
missing). Names starting with `_` or `.` are ignored. Restart the app to
load, update or remove an extension.

Git only tracks the built-in extensions (see below): anything else you put
in `extensions/` is ignored by `.gitignore`, so your own extensions never
show up as changes to commit.

## What it looks like in the app

- Tabs added with `add_tab` appear **before** the Settings tab.
- UI added with `add_ui` appears in an **Extensions** tab inside the
  generation tabs (txt2img, img2img, imgedit, any2video), one accordion per
  extension, and its values reach the hooks.
- When at least one extension exists, an **Extensions** tab **after**
  Settings lists each one: name, version, author, URL, description,
  dependencies and status (`loaded`, `skipped` or `failed`, with the
  reason).

## Describing an extension

Top-level constants in the entry file (all optional, except
`API_VERSION`, which is mandatory):

```python
NAME = 'My extension'          # shown in the Extensions tab
VERSION = '1.0'
AUTHOR = 'Someone'
URL = 'https://example.com/my-extension'
DESCRIPTION = 'What it does, in a sentence.'
REQUIRES = ['rembg[cpu]>=2.0,<3', 'pillow>=10']
HOSTS = ['example.com', '*.cdn.example.com']   # the sites it may contact
API_VERSION = 1                # REQUIRED: the API version it targets
PRIORITY = 0                   # lower loads first (tabs, hooks, installs)
```

`REQUIRES` and `HOSTS` can also be a dict that says **why** each entry is
needed. The reasons are shown to the user when the extension is approved
(and the package reasons again when it asks to install), and in the
Extensions tab; an entry without one shows "(no reason given)":

```python
REQUIRES = {
    'rembg[cpu]>=2.0,<3': 'removes the background of a photo',
    'pillow>=10': 'reads and saves the images',
}
HOSTS = {
    'example.com': 'the API it searches',
    '*.cdn.example.com': 'where example.com stores the files',
}
```

They are read **without importing the file** (they must be literals), so
the dependencies can be installed first and a broken extension can still
be described.

## Dependencies (`REQUIRES`)

Missing packages are installed **after you confirm** in the terminal, with
the interpreter that runs the app (progress is streamed to the terminal):

```
Extension 'Masker' requires missing package(s): rembg[cpu]>=2.0.87,<3. Install now? [y/N]
```

It uses `pip` when available, `uv pip` for uv-made environments (they have
no pip), and bootstraps pip with `ensurepip` otherwise. Nothing is ever
installed unconfirmed: answering anything but `y`, or running without a
terminal (a service, a pipe), skips the extension and says what to install.

- Entries are plain requirement strings and **every one needs a
  version**: a minimum (`'pkg>=1.0'`), a minimum and a maximum
  (`'pkg>=1.0,<2'`), a maximum (`'pkg<2'`) or an exact one
  (`'pkg==1.2'`), with extras if needed (`'pkg[extra]>=1.0'`). Without
  one the newest release would be installed, and a future one can break
  the extension. Pick the version you tested as the minimum and the next
  major release as the maximum. Entries without a version (`'pkg'`,
  `'pkg!=1.0'`), options (`--index-url=...`), paths and direct URLs
  (`pkg @ https://...`) make the extension fail to load, before it is
  asked about.
- A package counts as present when its distribution is installed at an
  accepted version — and so is everything its extras add: `rembg[cpu]` is
  *not* satisfied by a plain `rembg`.
- If you decline, the install fails, or no installer exists, the extension is skipped
  and the reason is printed in the terminal and shown in the Extensions tab; the rest
  load normally.

Only put extensions you trust in the folder: installing their
requirements and importing them runs code on your machine.

## The API

`register(api)` is called once per extension, with its own `api`:

```python
def register(api):
    api.add_tab('My Tab', render_tab)
    api.add_ui(render_options, tabs=('txt2img', 'img2img'))
    api.add_hook('pre_generation', on_pre)
    api.add_hook('post_generation', on_post)
    api.add_setting('level', 5)
```

| Method | Effect |
|---|---|
| `add_tab(title, render_fn)` | A new tab (CLI and server UIs). `render_fn()` builds its components. Duplicate titles are skipped. |
| `add_ui(render_fn, tabs=...)` | A section in the Extensions tab of the generation tabs (all four unless `tabs` lists some). `render_fn(tab)` builds components and returns `{name: component}` — the values the hooks need — or `None`. |
| `add_hook(name, fn)` | A `pre_generation`, `post_generation` or `replace_generation` hook (see Hooks). |
| `generate(params, tab='img2img', keep=False)` | Runs one generation of a tab through the app and waits for it: `params` is what you got in `ctx.params`, with what you need changed. Returns the images as PIL images; the files the app made for them are removed, and so are the command and prompt files the run adds to the history, unless `keep=True`: then the run stays as a normal generation of that tab (its files and history). Raises an error if the generation fails, and when the user pressed Stop (let it go: the app handles it). In CLI mode every call starts `sd-cli`, which loads the model each time; in server mode the model stays loaded. |
| `generate_live(params, tab='img2img', keep=False)` | `generate()` as a generator, for a `replace_generation` hook: `images = yield from api.generate_live(...)` shows the progress and the preview of that generation, as the tab does, and returns the images. With `keep=True` each image's `filename` is the file the app saved. |
| `api.Progress` | What a `replace_generation` generator can yield: `Progress(percent, text, image=None, sub=None)`. `image` goes to the results (`None` leaves the one shown), and `sub=(percent, text)` is a second progress bar under the main one. |
| `api.Stopped` | The error `generate()` and `generate_live()` raise when the user presses Stop: catch it to clean up or to show what you already have, and the Stop is yours to honour. |
| `api.param(params, name, default=None)` | In a hook: the value of one of your `add_ui` components. |
| `add_output(name='', label=None, gallery=True)` | Declares an output folder for what you generate, always inside `outputs/extensions/`: `outputs/extensions/<id>/`, or `outputs/extensions/<id>/<name>/` when you give a `name`. The extension only picks the name: it is sanitized to a single folder name (`../x` becomes `x`), and an extension can't choose any other location. Returns the path, created if missing. With `gallery=True` (images) the Gallery gets a button for the folder, named by `label` (default: `name`, else the extension's name). Use `gallery=False` for other files, such as `.txt` files with a refined prompt: the folder exists, the Gallery doesn't show it. Extensions that save nothing skip it. |
| `next_output_path(name='', extension='.png')` | The path of the next file of that output folder, numbered like the app's own outputs (`1.png`, `2.png`, ... or `1.txt`, `2.txt`, ...). `extension` must be a dot and up to 8 letters or digits (`.png`), anything else raises `ValueError`. Nothing is created: save to it. |
| `add_setting(key, default, label=None, info=None, choices=None, secret=False)` | Declares a setting (the name is letters, digits and underscores; it can't clash with an app setting or another extension's) (`bool`, `int`, `float` or `str`, keeping the type of `default`), stored as `ext_<id>_<key>` and set to `default` only when missing. It is editable in the Extensions tab. `secret=True` (text only) is for API keys: a password field that never shows the stored value (empty keeps it, a checkbox clears it); it is still saved as text in the config file. Returns the current value. |
| `get_setting(key)` | The current value of one of your settings (`KeyError` for any other key). |
| `set_setting(key, value)` | Changes a setting you declared (`KeyError` for any other key, `ValueError` for a value that doesn't fit). Returns the stored value. |
| `default_params()` | What a generation tab starts with, from the defaults the user saved: a base for the `params` of `generate()` that doesn't take the settings of another generation. |
| `serve(changes)` | In server mode, a generator for a `replace_generation` hook: `before = yield from api.serve({'in_cnnet': name})` makes sure the running sd-server has those launch params (the models and ControlNet the server loads when it starts), restarting it with them if it doesn't, and shows that it is. It returns what to give to `unserve()`, or None when nothing changed. It does nothing in CLI mode. Normal generations in other tabs wait while the server restarts. |
| `unserve(before)` | Put the server back as `serve()` found it. It waits and never raises; call it in a `finally`. |
| `models(kind)` | The installed models of a kind (the kinds of `download`), as the app lists them. |
| `choices(name)` | What sd.cpp accepts for `'samplers'`, `'schedulers'`, `'rng'`, `'sampler_rng'`, `'prediction'`, `'previews'` or `'control_types'`. |
| `data_dir()` | Your own folder for what you keep between runs (models, caches): `user_data/extensions/<id>/`, next to the config file, created if missing. |
| `add_gallery_action(label, fn)` | A button in the Gallery running `fn(ctx)` on the selected **image** (`ctx.path`; never videos). Return a message to show, or `None`. |
| `fetch_json(url, headers=None)` | GET a url and return the parsed JSON (5 MB at most). See *Network access* below. |
| `download(url, filename, kind=None, headers=None, sha256=None, progress=None)` | Download a file. With a `kind` it goes to the app's folder for that kind of model (`checkpoint`, `diffusion`, `vae`, `text_encoder`, `embedding`, `lora`, `upscaler`, `controlnet`, `adetailer`, `ipadapter`, `pulid`, `animatediff`, `photomaker`, `taesd`) and the extension never learns the path; without one it goes to `data_dir()`. `filename` is a plain name (model files end in `.safetensors`, `.sft`, `.gguf`, `.ckpt`, `.pt`, `.pth`, `.bin` or `.onnx`); an existing file is never replaced (`FileExistsError`). `progress(done, total)` is called while downloading (raise from it to cancel); `sha256` is checked when given. The file appears complete or not at all. Returns `{'name', 'size', 'sha256', 'host'}`. |
| `notify(message, level='info')` | A toast in the page when called from a UI event (`level`: `info` or `warning`); otherwise it is only printed in the terminal. |
| `api.id` / `api.version` | Your id (file or folder name, key-safe) / the API version you declared. |

If `register` (or the import) raises, the extension is marked `failed`
and nothing it registered stays active.

## Hooks

Hooks get a context object `ctx`:

- `pre_generation` — `fn(ctx)`: called in the worker thread, just before
  the sdcpp call. Change `ctx.params` in place (e.g. add a prompt
  suffix); your `add_ui` values are in it. The prompts are
  `ctx.params['in_pprompt']` and `ctx.params['in_nprompt']`. `ctx.tab` is
  the generation tab (`txt2img`, `img2img`, `imgedit`, `any2video`).
- `post_generation` — `fn(ctx)`: called after generation. `ctx.images`
  holds the final images — a list of path strings in CLI mode, a list of
  PIL images in server mode, or `None` when nothing was produced. To
  replace them, assign `ctx.images = [...]` (or change the list in
  place). Return values are ignored.
- `replace_generation` — `fn(ctx)`: return `None` to let the app generate,
  or a generator that generates **instead**. The generator yields
  `(percent, text)` to show progress (or `(percent, text, image)` to show
  an image in the results while it works), or `api.Progress(percent, text,
  image=None, sub=None)`, where `sub=(percent, text)` adds a second, thinner
  progress bar under the main one (for the part of the work in progress),
  and returns the final images (paths or PIL images); inside it, `api.generate(...)` runs generations through
  the app, so the model, sampler and every other setting of the tab are
  yours without rebuilding them. The first extension that takes over wins;
  post hooks still run on its images. If it raises, the user sees the
  error, and the Stop button ends it (`api.generate` raises when pressed).

Hooks run in extension `PRIORITY` order (then id). Errors are caught
and printed with their traceback; generation continues. Hooks apply to the generation tabs,
not to Checkpoint Converter or Upscaler.

## Network access

Extensions reach the network through the app (`fetch_json`, `download`),
and only the hosts they declare in `HOSTS`: names like `example.com` or
`*.example.com` (a wildcard matches subdomains only). Declare the sites
and the storage they redirect to; the Extensions tab shows the list. The
app:

- allows `https` only, on the default port, without credentials in the
  url, and never to an address that isn't public (no `localhost`, no
  `192.168.x.x`);
- follows redirects itself and checks every hop against `HOSTS`; the
  `headers` (an API key) are sent to the first host only, never to where
  it redirects;
- puts downloads in a folder it picks, never one the extension names.

Every download, finished, failed or refused, is added to a permanent log
(`user_data/extensions/downloads.jsonl`, shown in the **Downloads** table
of the Extensions tab): time, extension, kind, file, host, size, SHA256
and where it was saved. The query part of the url (where tokens live) and
the headers are never logged. An extension can't erase the log through the
api.

There is no call to refresh the model lists: they are read from the disk
every time, so a downloaded model appears when you press the refresh
button next to the list.

These checks protect an extension from hostile data (a redirect, a url or
a file name that comes from a server). They are not a sandbox: an
extension is Python code, and one that opens its own connections instead
of using the api isn't covered (nor logged).

`HOSTS` is a declaration, not a restriction the app can enforce: it says
what the extension intends to contact, so you can see it in the Extensions
tab and the app can refuse a redirect somewhere else. An extension can
still reach any host by other means.

## Trust

An extension is Python code that runs with all your permissions, and so
are the packages it lists in `REQUIRES`. Nothing in the api (the version
views, the network checks, the download log) stops a malicious extension.
Only install extensions from people you trust.

The first time the app sees an extension it asks in the terminal, before
importing or installing anything. It shows the name, version, author,
URL, description, where the files are, a short SHA256, the `HOSTS` it
declares, the folder its images go to (`outputs/extensions/<id>/`) and the
packages it requires, then `Load this extension? [y/N]`.

- **y**: the SHA256 of all its files is saved in
  `user_data/extensions/trusted.json`. It loads without asking again, until
  a file changes (an update): then it asks again and shows "Updated extension" instead of "New extension".
- **n** (or Enter): it doesn't load. Nothing is remembered and the files
  stay where they are, so it asks again the next time the app starts.
- **No terminal** (a service, a pipe): nobody can answer, so new and
  changed extensions don't load. Start the app once from a terminal.

At every start the app removes the approvals of extensions that are no
longer in the folder, so one put back later asks again. To forget an
approval by hand, delete its entry in `trusted.json`. Missing
packages are asked separately, after the extension is approved, with a
warning about running package code.

## API versions

Every extension **must** declare `API_VERSION`; without it, or with a
version that doesn't exist, it fails to load. An extension targeting a
newer version than the app has is skipped.

The app has API v1 today. If a later change breaks what extensions
already use, it becomes a new version and extensions that declare the old
number keep the old behaviour. `api.version` is the one you declared.

## The bundled extensions

The `extensions/` folder ships four, which are also the examples to read
(each asks for your approval the first time, like any extension):

- **Civitai** (`extensions/civitai/`): search and download models.
- **Masker** (`extensions/masker.py`): foreground masks.
- **Pose** (`extensions/pose/`): OpenPose skeletons for ControlNet.
- **Ultimate SDCPP Upscale** (`extensions/ultimate_sdcpp_upscale/`): upscales what a
  generation makes, tile by tile (a `replace_generation` hook that uses
  `api.generate_live()` and, in server mode, `api.serve()` for the ControlNet).
  Based on the ideas of [ComfyUI_UltimateSDUpscale](https://github.com/ssitu/ComfyUI_UltimateSDUpscale) and
  [ultimate-upscale-for-automatic1111](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111).

## Example: UI injected into the generation tabs

```python
NAME = 'Prompt suffix'
API_VERSION = 1

def render(tab):
    import gradio as gr
    return {'suffix': gr.Textbox(label=f'Suffix for {tab}')}

def register(api):
    api.add_ui(render)
    api.add_hook(
        'pre_generation',
        lambda ctx: ctx.params.update(
            in_pprompt=(ctx.params.get('in_pprompt') or '')
            + api.param(ctx.params, 'suffix', '')
        )
    )
```

## Example: saving text instead of images

An output folder doesn't have to hold images. With `gallery=False` the
folder is created but the Gallery gets no button for it:

```python
NAME = 'Prompt log'
API_VERSION = 1

def register(api):
    api.add_output('prompts', gallery=False)

    def log(ctx):
        path = api.next_output_path('prompts', '.txt')   # 1.txt, 2.txt...
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(ctx.params.get('in_pprompt') or ''))

    api.add_hook('pre_generation', log)
```

A later hook (or a tab of your own) can read those files and put them back
in `in_pprompt` or `in_nprompt`.

## Example: `extensions/civitai/`

Search and download models from civitai.com or its civitai.red mirror:
search by name or paste a model or version link, filter by type and base
model (SDXL 1.0, Anima, Qwen 2.1...; new base models are learned from the
results and offered too), browse a scrollable gallery of models with
cover pictures (with *Previous page* and *Next page*), pick a version (its previews show in a gallery; click one to see its prompt,
negative prompt, settings and resources, read from Civitai's metadata and
ComfyUI workflows) and a file, and the app
saves it in the model folder of its kind (by the file's own type, or the
folder you pick in *Save to*). It declares `HOSTS`
for both sites and their storage, keeps the API key as a `secret` setting,
verifies each file against Civitai's SHA256, and only offers model files
that weren't flagged by Civitai's scans. It is **safe for work by
default**: the *Show NSFW models and previews* setting is off, so the
search asks Civitai for safe models only, models flagged NSFW are dropped
from the answer (and a pasted link to one says why it isn't shown), and
only pictures Civitai rates safe are loaded, so a picture with no rating
is treated as not safe.

## Example: `extensions/masker.py`

Declares `REQUIRES = {'rembg[cpu]>=2.0.87,<3': ...}` (offered for install on first start) and adds
a 'Masker' tab that creates a foreground mask from a photo with rembg,
saves it to its output folder (`outputs/extensions/masker/`, via `add_output()`, so
the masks get their own Gallery button), and can later
apply a saved mask to any other image — a mask is a reusable artifact,
not a one-off. Masks are plain PNGs: load one in the img2img Mask box.

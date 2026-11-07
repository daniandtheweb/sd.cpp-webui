"""sd.cpp-webui - extensions - the API an extension calls

register(api) gets its own api object (an ExtensionAPI). Everything it
adds is recorded on its Extension (host.py), so a register() that fails
halfway leaves nothing behind. The API version is in versions.py.
"""

import os
import re
from collections.abc import Callable

from modules.extensions import net, versions
from modules.extensions.host import (
    Extension,
    SettingSpec,
    downloads_log_path,
    extensions,
    make_id,
    set_setting_value,
    setting_value,
)


def _config():
    # Imported here: shared_instance starts the app's workers, which
    # the extensions package must not do just by being imported.
    from modules.shared_instance import config
    return config


# A file extension: a dot and a few letters or digits, nothing else
_SETTING_KEY = re.compile(r'[A-Za-z0-9_]+')
_FILE_EXTENSION = re.compile(r'\.[A-Za-z0-9]{1,8}')


def _need_callable(fn: Callable, what: str):
    if not callable(fn):
        raise TypeError(f"{what} must be callable")


class ExtensionAPI:
    """The implementation behind the api object of an extension."""

    def __init__(self, ext: Extension):
        self._ext = ext

    @property
    def id(self) -> str:
        """The extension's id (its file or folder name, key-safe)."""
        return self._ext.id

    @property
    def version(self) -> int:
        """The API version the extension declared."""
        return self._ext.api_version

    # -- User interface ----------------------------------------------------

    def add_tab(self, title: str, render_fn: Callable):
        """
        Add a tab, before the Settings tab. render_fn() builds its
        components. Duplicate titles are skipped with a warning.
        """
        _need_callable(render_fn, 'render_fn')
        self._ext.tabs.append((str(title), render_fn))

    def add_ui(self, render_fn: Callable, tabs: tuple[str, ...] | None = None):
        """
        Add a section to the 'Extensions' tab inside the generation tabs
        (all of them unless tabs says which). render_fn(tab) builds the
        components and returns a dict {name: component} of the values the
        hooks need, or None. Read them in a hook with
        api.param(ctx.params, name).
        """
        _need_callable(render_fn, 'render_fn')
        valid = versions.TABS
        if tabs is None:
            tabs = valid
        elif isinstance(tabs, str):
            tabs = (tabs,)
        for tab in tabs:
            if tab not in valid:
                raise ValueError(
                    f'unknown tab {tab!r} (valid: {", ".join(valid)})'
                )
        self._ext.uis.append((tuple(tabs), render_fn))

    def param(self, params: dict, name: str, default=None):
        """A value of this extension's add_ui components, in a hook."""
        return params.get(self._ext.key(name), default)

    def add_gallery_action(self, label: str, fn: Callable):
        """
        Add a button to the Gallery that runs fn(ctx) on the selected
        image (ctx.path). It returns a message to show, or None. It only
        runs for images, never for videos.
        """
        _need_callable(fn, "gallery action")
        if not str(label).strip():
            raise ValueError("a gallery action needs a label")
        self._ext.gallery_actions.append((str(label).strip(), fn))

    # -- Hooks -------------------------------------------------------------

    def add_hook(self, name: str, fn: Callable):
        """
        Register a generation hook: fn(ctx). pre_generation hooks change
        ctx.params. post_generation hooks read ctx.params and may replace
        ctx.images. ctx.tab is the generation tab.
        """
        valid = versions.HOOKS
        if name not in valid:
            raise ValueError(
                f'unknown hook {name!r} (valid: {", ".join(valid)})'
            )
        _need_callable(fn, "hook fn")
        self._ext.hooks.setdefault(name, []).append(fn)

    # -- Settings ----------------------------------------------------------

    def add_setting(
        self, key: str, default, label: str | None = None,
        info: str | None = None, choices=None, secret: bool = False
    ):
        """
        Declare a setting of the extension: stored in the config as
        ext_<id>_<key>, set to default only when missing, and editable
        in the Extensions tab (label, info and choices describe it
        there). The default is a bool, int, float or str, and the setting
        keeps that type. secret=True (text only) is for keys and tokens:
        the Extensions tab shows a password field that never displays the
        stored value. It is still saved in the config file as text.
        Returns the current value.
        """
        if not _SETTING_KEY.fullmatch(str(key)):
            raise ValueError(
                f"{key!r} is not a setting name: use letters, digits "
                "and underscores"
            )
        self._check_config_key(key)
        if type(default) not in (bool, int, float, str):
            raise TypeError("a setting is a bool, int, float or str")
        if secret and type(default) is not str:
            raise TypeError("only a text setting can be secret")
        spec = SettingSpec(
            key, default, label or key.replace('_', " ").capitalize(),
            info or '', tuple(choices or ()), secret
        )
        if spec.choices and default not in spec.choices:
            raise ValueError(f"the default of {key!r} is not one of choices")
        self._ext.settings[key] = spec

        config = _config()
        if self._ext.key(key) not in config.data:
            config.data[self._ext.key(key)] = default
            config.save_config()
        return setting_value(self._ext, key)

    def _check_config_key(self, key: str):
        """
        The config key of a setting must be this extension's alone: not an
        app setting, not another extension's (ext_a_b + c and ext_a +
        b_c are the same key).
        """
        from modules.settings.defaults import DEFAULT_SETTINGS

        full = self._ext.key(key)
        if full in DEFAULT_SETTINGS:
            raise ValueError(f"{full!r} is a setting of the app")
        for other in extensions():
            if other is not self._ext and any(
                other.key(name) == full for name in other.settings
            ):
                raise ValueError(
                    f"{full!r} is already a setting of {other.id!r}"
                )

    def get_setting(self, key: str):
        """
        The current value of a setting the extension declared with
        add_setting (its default when the stored one isn't valid).
        Raises KeyError for any other key.
        """
        return setting_value(self._ext, key)

    def set_setting(self, key: str, value):
        """
        Change a setting the extension declared with add_setting. Raises
        KeyError for any other key and ValueError for a value that
        doesn't fit the setting. Returns the value as stored.
        """
        return set_setting_value(self._ext, key, value)

    # -- Folders -----------------------------------------------------------

    def data_dir(self) -> str:
        """
        The extension's own folder for what it keeps between runs (models
        it downloads, caches): user_data/extensions/<id>/, next to the
        config file, created if missing. It is the same for every run and
        it is not where the extension's code lives.
        """
        base = os.path.dirname(os.path.abspath(_config().config_path))
        path = os.path.join(base, 'extensions', self._ext.id)
        os.makedirs(path, exist_ok=True)
        return path

    # -- Network -----------------------------------------------------------

    def fetch_json(self, url: str, headers: dict | None = None):
        """
        GET a url and return the parsed JSON. Only https, only to the hosts
        the extension declared with HOSTS (redirects included). headers
        (an API key) go to the first host only.
        """
        return net.fetch_json(url, self._ext.hosts, headers)

    def download(
        self, url: str, filename: str, kind: str | None = None,
        headers: dict | None = None, sha256: str | None = None,
        progress: Callable | None = None
    ) -> dict:
        """
        Download a file and save it as filename: in the app's folder for
        that kind of model (kind: checkpoint, diffusion, vae,
        text_encoder, embedding, lora, upscaler, controlnet, adetailer,
        ipadapter, pulid, animatediff, photomaker or taesd), or in the
        extension's own data_dir() when kind is None. The extension never
        learns the model folder, and it never replaces a file that exists
        (FileExistsError). Only https, only to the hosts declared with
        HOSTS. progress(done, total) is called while it downloads (total
        may be None; raising from it cancels). sha256 is checked when
        given. Every download, finished or not, is written to the
        permanent log in the Extensions tab.
        Returns {'name', 'size', 'sha256', 'host'}.
        """
        model = kind is not None
        if model and kind not in net.MODEL_KINDS:
            raise ValueError(
                f"unknown kind {kind!r} "
                f'(valid: {", ".join(net.MODEL_KINDS)})'
            )
        name = net.safe_filename(filename, model)
        if model:
            # data, not get(): get() hides a folder that doesn't exist yet
            folder = _config().data.get(net.MODEL_KINDS[kind])
            if not folder:
                raise ValueError(f"no folder is set for {kind} models")
            folder = os.path.abspath(folder)
        else:
            folder = self.data_dir()

        record = {
            'extension': self._ext.id, 'kind': kind or 'data', 'file': name,
            'saved_to': os.path.join(folder, name), 'url': str(url),
        }
        try:
            result = net.download(
                url, self._ext.hosts, folder, name, headers, sha256, progress
            )
        except BaseException as e:
            self._log(status='failed', error=f"{type(e).__name__}: {e}",
                      **record)
            raise
        self._log(status='ok', bytes=result['size'],
                  sha256=result['sha256'], host=result['host'], **record)
        return result

    @staticmethod
    def _log(**record):
        try:
            net.log_download(downloads_log_path(), **record)
        except OSError as e:
            print(f"Could not write the downloads log: {e}")

    def notify(self, message: str, level: str = 'info'):
        """
        Show a message to the user (a toast in the page) when called from
        an event of the UI; otherwise it only goes to the log. level:
        'info' or 'warning'.
        """
        if level not in ('info', 'warning'):
            raise ValueError("level is 'info' or 'warning'")
        from gradio.context import LocalContext

        in_event = (LocalContext.blocks.get(None) is not None
                    and LocalContext.event_id.get(None) is not None)
        if not in_event:
            print(f"{self._ext.id}: {message}")
            return
        import gradio as gr
        (gr.Info if level == 'info' else gr.Warning)(str(message))

    def _output_dir(self, name: str) -> str:
        """
        outputs/extensions/<id>/ or .../<id>/<name>/. The extension only
        picks the name, and make_id turns it into a single safe folder
        name ('../x' becomes 'x'), so it can't point anywhere else.
        """
        base = self._ext.output_dir
        return os.path.join(base, make_id(name), '') if name else base

    def add_output(
        self, name: str = '', label: str | None = None,
        gallery: bool = True
    ) -> str:
        """
        Declare an output folder for what this extension generates. It is
        always inside outputs/extensions/: outputs/extensions/<id>/ (or
        .../<id>/<name>/ when a name is given; it is sanitized to a single
        folder name). Returns the folder's path, created if missing.
        With gallery=True (images) the Gallery gets a button for it, named
        by label (the name, or the extension's name). Pass gallery=False
        for anything else, such as .txt files with a refined prompt: the
        folder exists but the Gallery shows nothing for it.
        Extensions that save nothing don't need this.
        """
        path = self._output_dir(name)
        os.makedirs(path, exist_ok=True)
        if gallery:
            self._ext.outputs[path] = label or name or self._ext.name
        return path

    def next_output_path(
        self, name: str = '', extension: str = '.png'
    ) -> str:
        """
        The path for the next file of an output folder declared with
        add_output(name), numbered the way the app names its own
        outputs (1.png, 2.png, ... or 1.txt, 2.txt, ...). Nothing is
        created: save to it.
        """
        from modules.gallery import next_media_name

        if not _FILE_EXTENSION.fullmatch(str(extension)):
            raise ValueError(
                f'extension must look like ".png", not {extension!r}'
            )
        folder = self._output_dir(name)
        return os.path.join(folder, next_media_name(folder, extension))


def api_for(ext: Extension) -> ExtensionAPI:
    """The api object of an extension."""
    return ExtensionAPI(ext)

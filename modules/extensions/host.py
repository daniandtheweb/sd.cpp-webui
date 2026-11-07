"""sd.cpp-webui - extensions - what the app knows about its extensions

The loader records every extension it finds here (an Extension), and the
UIs and the generation pipeline read it back lazily, at call time. Only
the extensions that loaded successfully count. The part extensions call
is in api.py, and the API version in versions.py.
"""

import os
import re
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MappingProxyType

from modules.extensions.meta import Meta
from modules.extensions.versions import HOOKS as VALID_HOOKS

LOADED, SKIPPED, FAILED = 'loaded', 'skipped', 'failed'

# What a gallery action can be given: the gallery also holds videos
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp')


def report_error(message: str):
    """
    Print what went wrong in an extension's code, from the except block
    that caught it: the message and the error, then the traceback an
    extension author needs.
    """
    error = sys.exc_info()[1]
    print(f"{message}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _config():
    # Imported here: shared_instance starts the app's workers, which
    # the extensions package must not do just by being imported.
    from modules.shared_instance import config
    return config


@dataclass
class SettingSpec:
    """A setting an extension declared: what the UI shows to edit it."""
    key: str
    default: object
    label: str = ''
    info: str = ''
    choices: tuple = ()
    # Shown as dots, never displayed back, kept when left empty
    secret: bool = False

    def coerce(self, value):
        """The value as the type of the default, or ValueError."""
        kind = type(self.default)
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{self.key}: expected true or false")
        elif kind is int:
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or int(value) != value):
                raise ValueError(f"{self.key}: expected a whole number")
            value = int(value)
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.key}: expected a number")
            value = float(value)
        elif not isinstance(value, str):
            raise ValueError(f"{self.key}: expected text")
        if self.choices and value not in self.choices:
            raise ValueError(
                f"{self.key}: {value!r} is not one of {list(self.choices)}"
            )
        return value


@dataclass
class Extension:
    """One discovered extension: what it says, how it went, what it added."""
    id: str
    path: str
    meta: Meta
    status: str = SKIPPED
    reason: str = ''

    # What register(api) added
    tabs: list[tuple[str, Callable]] = field(default_factory=list)
    uis: list[tuple[tuple[str, ...], Callable]] = field(default_factory=list)
    hooks: dict[str, list[Callable]] = field(
        default_factory=lambda: {name: [] for name in VALID_HOOKS}
    )
    settings: dict[str, SettingSpec] = field(default_factory=dict)
    gallery_actions: list[tuple[str, Callable]] = field(default_factory=list)
    # folder -> label, of the output folders it declared
    outputs: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.meta.name or self.id

    @property
    def api_version(self) -> int | None:
        return self.meta.api_version

    @property
    def output_dir(self) -> str:
        """Where all its images go: outputs/extensions/<id>/ (the app's)."""
        from modules.settings.defaults import CURRENT_DIR
        return os.path.join(CURRENT_DIR, 'outputs', 'extensions', self.id, '')

    @property
    def hosts(self) -> list[str]:
        """The hosts it declared with HOSTS: all it may contact."""
        return self.meta.hosts

    @property
    def sort_key(self) -> tuple:
        """Extensions run by priority (lower first), then by id."""
        return (self.meta.priority, self.id)

    def key(self, name: str) -> str:
        """
        A key of this extension, in the config and in the params:
        ext_<id>_<name>.
        """
        return f'ext_{self.id}_{name}'

    def skip(self, reason: str):
        """Not loaded, but nothing is wrong with the extension itself."""
        self.status, self.reason = SKIPPED, reason

    def fail(self, reason: str):
        """Not loaded: it broke."""
        self.status, self.reason = FAILED, reason

    def clear(self):
        """Forget everything register(api) added."""
        self.tabs.clear()
        self.uis.clear()
        self.settings.clear()
        self.gallery_actions.clear()
        self.outputs.clear()
        for fns in self.hooks.values():
            fns.clear()


class HookContext:
    """What a generation hook is given."""

    def __init__(self, tab: str, params: dict, images=None):
        self.tab = tab
        self.params = params
        self.images = images


class GalleryContext:
    """What a gallery action is given."""

    def __init__(self, path: str):
        self.path = path


_EXTENSIONS: list[Extension] = []


def make_id(name: str) -> str:
    """A key-safe id from a file or folder name."""
    return re.sub(r'\W+', '_', name).strip('_').lower() or 'ext'


def add_extension(ext: Extension):
    _EXTENSIONS.append(ext)


def extensions() -> list[Extension]:
    """Every discovered extension, whatever its status."""
    return list(_EXTENSIONS)


def _active() -> list[Extension]:
    """The loaded extensions, in the order they run."""
    return sorted(
        (ext for ext in _EXTENSIONS if ext.status == LOADED),
        key=lambda ext: ext.sort_key
    )


def registered_tabs() -> list[tuple[str, Callable]]:
    """(title, render_fn) of the tabs extensions add."""
    return [tab for ext in _active() for tab in ext.tabs]


def registered_uis(tab: str) -> list[tuple[Extension, Callable]]:
    """(extension, render_fn) of the UIs added to a generation tab."""
    return [
        (ext, render)
        for ext in _active()
        for tabs, render in ext.uis
        if tab in tabs
    ]


def registered_outputs() -> list[tuple[Extension, str, str]]:
    """(extension, label, folder) of the output folders declared."""
    return [
        (ext, label, folder)
        for ext in _active()
        for folder, label in ext.outputs.items()
    ]


def registered_gallery_actions() -> list[tuple[Extension, str, Callable]]:
    """(extension, label, fn) of the actions added to the Gallery."""
    return [
        (ext, label, fn)
        for ext in _active()
        for label, fn in ext.gallery_actions
    ]


def registered_settings() -> list[Extension]:
    """The loaded extensions that declared settings."""
    return [ext for ext in _active() if ext.settings]


def generation_tab(func_name: str) -> str:
    """The tab of a generation function: txt2img_api -> txt2img."""
    return func_name.removesuffix('_api')


# -- Downloads -------------------------------------------------------------

def downloads_log_path() -> str:
    from modules.extensions.net import log_path
    return log_path(_config().config_path)


def downloads_log() -> list[dict]:
    """What extensions downloaded, newest first (it is never erased)."""
    from modules.extensions.net import read_downloads
    return read_downloads(downloads_log_path())


# -- Settings --------------------------------------------------------------

def setting_value(ext: Extension, key: str):
    """
    The current value of a declared setting; the default when the stored
    one is not valid (for example edited by hand in the config).
    """
    spec = ext.settings[key]
    stored = _config().data.get(ext.key(key), spec.default)
    try:
        return spec.coerce(stored)
    except ValueError:
        return spec.default


def set_setting_value(ext: Extension, key: str, value):
    """
    Store a new value of a declared setting; returns it as stored.
    Raises KeyError for a setting the extension didn't declare and
    ValueError for a value that doesn't fit it.
    """
    if key not in ext.settings:
        raise KeyError(f"{key!r} is not a setting of {ext.id}")
    value = ext.settings[key].coerce(value)
    config = _config()
    config.data[ext.key(key)] = value
    config.save_config()
    return value


# -- Hooks -----------------------------------------------------------------

def _hooks(name: str):
    """(extension, fn) of the hooks called `name`, in running order."""
    for ext in _active():
        for fn in ext.hooks.get(name, ()):
            yield ext, fn


def _fn_name(fn) -> str:
    return getattr(fn, '__name__', repr(fn))


def run_pre(params: dict, tab: str = ''):
    """
    Run the pre_generation hooks; each may change ctx.params.
    """
    for ext, fn in _hooks('pre_generation'):
        try:
            fn(HookContext(tab, params))
        except (Exception, SystemExit):
            report_error(
                f"{ext.id}: pre_generation hook {_fn_name(fn)} failed"
            )


def _same_items(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))


def run_post(images, params: dict, tab: str = ''):
    """
    Chain the post_generation hooks, which see the images in ctx.images and
    replace them by assigning it. Returns the final images, or None when
    no hook changed them.
    """
    # Post hooks read what the generation was built from, not change it
    ctx = HookContext(tab, MappingProxyType(params), images)
    before = None if images is None else list(images)
    for ext, fn in _hooks('post_generation'):
        try:
            fn(ctx)
        except (Exception, SystemExit):
            report_error(
                f"{ext.id}: post_generation hook {_fn_name(fn)} failed"
            )
    return None if _same_items(before, ctx.images) else ctx.images


# -- Gallery actions -------------------------------------------------------

def run_gallery_action(ext: Extension, fn: Callable, path) -> str:
    """
    Run an extension's gallery action on the selected file; returns the
    message to show: what the action returned, or why it didn't run.
    """
    if not path or not os.path.isfile(path):
        return "Select an image first."
    if not str(path).lower().endswith(IMAGE_EXTENSIONS):
        return "Select an image: the action works on images only."
    try:
        result = fn(GalleryContext(path))
    except (Exception, SystemExit) as e:
        report_error(f"{ext.id}: gallery action {_fn_name(fn)} failed")
        return f"Failed: {type(e).__name__}: {e}"
    return str(result) if result else 'Done.'


def _reset():
    """Forget every extension (for tests)."""
    _EXTENSIONS.clear()

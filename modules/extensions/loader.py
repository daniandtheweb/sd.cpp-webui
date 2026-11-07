"""sd.cpp-webui - extensions - the loader

Finds the extensions of the configured ext_dir, asks before running new or
changed ones (trust.py), asks to install what they require, imports them
and calls their register(api). A failing extension is reported and
skipped; it never stops the rest or the app.

An extension is either a single file (ext_dir/name.py) or a folder
(ext_dir/name/extension.py, next to its own helper modules). Names
starting with '_' or '.' are ignored.
"""

import importlib.util
import os
import sys

from modules.extensions import deps, trust
from modules.extensions.api import api_for
from modules.extensions.host import (
    FAILED,
    LOADED,
    SKIPPED,
    Extension,
    add_extension,
    extensions,
    make_id,
)
from modules.extensions.meta import read_meta
from modules.extensions.versions import LATEST
from modules.settings.defaults import DIR_DEFAULTS

ENTRY_FILE = 'extension.py'

# Fallback for a missing ext_dir: the folder the config defaults to (they
# resolve it from the app's own location, not from the CWD)
_DEFAULT_EXT_DIR = os.path.normpath(DIR_DEFAULTS['ext_dir'])

# Extensions are opt-in: without --extensions nothing is discovered,
# installed or imported (read from argv like the mode in shared_instance,
# because the tab modules need the answer while they are imported).
def _flag_given(argv: list[str]) -> bool:
    """
    Whether --extensions is in argv, spelled the way argparse accepts it
    (any unique prefix: --ext, --extens...), so the flag can't be
    accepted by the parser and ignored here.
    """
    return any(
        len(arg) > 2 and '--extensions'.startswith(arg)
        for arg in argv[1:] if arg.startswith('--')
    )


ENABLED = _flag_given(sys.argv)

_loaded = False


def _loaded_ids() -> list[str]:
    return [ext.id for ext in extensions() if ext.status == LOADED]


def load_extensions(config) -> list[str]:
    """
    Load the extensions of config's ext_dir, once; return the ids of
    the loaded ones. A later call returns the same result: the tab
    modules ask for the extensions while they are imported, before
    the app's main() runs. Does nothing unless --extensions was given.
    """
    global _loaded
    if _loaded:
        return _loaded_ids()
    _loaded = True
    if not ENABLED:
        return []

    ext_dir = _resolve_dir(config.data.get('ext_dir'))
    if ext_dir is None:
        return []

    found = _discover(ext_dir)
    if found is None:  # the folder could not be listed: forget nothing
        return []
    trust.forget_missing({ext.id for ext in found})

    for ext in sorted(found, key=lambda e: e.sort_key):
        add_extension(ext)
        _load(ext)

    print(_report(ext_dir))
    return _loaded_ids()


# -- Finding the extensions ------------------------------------------------

def _resolve_dir(raw) -> str | None:
    """The extensions folder to use, created when it is the default."""
    if raw and not os.path.isdir(raw):
        # Stale value (e.g. a CWD-based path from an old config):
        # fall back to the repo default instead of loading nothing.
        print(f"ext_dir {raw} missing; using {_DEFAULT_EXT_DIR}")
        raw = None
    if raw:
        return raw
    try:
        os.makedirs(_DEFAULT_EXT_DIR, exist_ok=True)
    except OSError as e:
        print(
            f"Loaded 0 extensions (cannot create {_DEFAULT_EXT_DIR}: {e})"
        )
        return None
    return _DEFAULT_EXT_DIR


def _entry(ext_dir: str, name: str) -> tuple[str, str] | None:
    """(entry file, name without extension) of ext_dir/name, if it is one."""
    path = os.path.join(ext_dir, name)
    if os.path.isdir(path):
        entry = os.path.join(path, ENTRY_FILE)
        return (entry, name) if os.path.isfile(entry) else None
    if name.endswith('.py') and name != ENTRY_FILE:  # that is a folder's
        return path, name[:-3]
    return None


def _discover(ext_dir: str) -> list[Extension] | None:
    """
    Read the metadata of every extension in ext_dir (nothing imported);
    None when the folder can't be listed.
    """
    try:
        names = sorted(os.listdir(ext_dir))
    except OSError as e:
        print(f"Loaded 0 extensions (cannot list {ext_dir}: {e})")
        return None

    found, ids = [], set()
    for name in names:
        entry = None if name.startswith(('_', '.')) else _entry(ext_dir, name)
        if entry is None:
            continue
        path, stem = entry

        ext = Extension(
            id=make_id(stem), path=path, meta=read_meta(path, stem)
        )
        for problem in ext.meta.problems:
            print(f"{ext.id}: {problem}")
        if ext.id in ids:
            ext.fail(f"duplicate id {ext.id!r}")
        ids.add(ext.id)
        found.append(ext)
    return found


# -- Loading one extension -------------------------------------------------

def _load(ext: Extension):
    """Install, import and register one extension; record the outcome."""
    if ext.status == FAILED:  # discovery already rejected it
        return

    version = ext.meta.api_version
    if version is None:
        ext.fail(
            "no API_VERSION: declare the API version it targets, "
            f"for example API_VERSION = {LATEST}"
        )
        return
    if version > LATEST:
        ext.skip(
            f"targets API v{version}, this app supports up to v{LATEST}"
        )
        return
    if version < 1:
        ext.fail(f"API v{version} does not exist (the first is v1)")
        return

    _, bad = deps.parse_requires(ext.meta.requires)
    if bad:  # no point asking about an extension that can't be installed
        ext.fail(deps.invalid_message(bad))
        return

    # Before anything is installed or run: the user decides on new code
    error = trust.approve(ext, _is_folder(ext))
    if error:
        ext.skip(error)
        return

    error = deps.ensure(
        ext.meta.requires,
        lambda reqs: deps.ask_install(
            ext.name, reqs, ext.meta.require_reasons)
    )
    if error:
        ext.skip(f"dependencies: {error}")
        return

    try:
        module = _import(ext)
    except (Exception, SystemExit) as e:
        # SystemExit too: a stray sys.exit() in an extension (or in a
        # library it imports) must not abort boot.
        ext.fail(f"import: {type(e).__name__}: {e}")
        return

    register = getattr(module, 'register', None)
    if not callable(register):
        ext.fail("no register(api) function")
        return

    try:
        register(api_for(ext))
    except (Exception, SystemExit) as e:
        ext.clear()
        ext.fail(f"register: {type(e).__name__}: {e}")
        return

    ext.status = LOADED


def _is_folder(ext: Extension) -> bool:
    return os.path.basename(ext.path) == ENTRY_FILE


def _import(ext: Extension):
    """
    Import the entry file as sdcpp_ext_<id>. A folder extension is a
    package, so its own modules import with `from . import helper`.
    """
    module_name = f'sdcpp_ext_{ext.id}'
    is_folder = _is_folder(ext)
    spec = importlib.util.spec_from_file_location(
        module_name, ext.path,
        submodule_search_locations=(
            [os.path.dirname(ext.path)] if is_folder else None
        )
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ext.path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Forget the half-imported extension and its submodules
        for key in [k for k in sys.modules
                    if k == module_name or k.startswith(module_name + '.')]:
            del sys.modules[key]
        raise
    return module


# -- The startup report ----------------------------------------------------

def _describe(ext: Extension) -> str:
    """Name and version: 'Masker v1.0'."""
    version = ext.meta.version
    if version and version[:1] not in 'vV':
        version = f'v{version}'
    return f"{ext.name} {version}".strip()


def _block(title: str, exts: list[Extension]) -> str:
    """One group of the report: a title, then a line per extension."""
    count = f"{len(exts)} extension" + ('' if len(exts) == 1 else 's')
    if not exts:
        return title.format(count)

    lines = [title.format(count) + ':', '']
    for ext in exts:
        line = f"  - {_describe(ext)}"
        if ext.status != LOADED:
            line += f": {ext.reason}"
        lines.append(line)
    return '\n'.join(lines)


def _report(ext_dir: str) -> str:
    """The startup summary: which extensions loaded and which did not."""
    blocks = []
    for status, title in (
        (LOADED, "Loaded {} from " + ext_dir),
        (SKIPPED, "Skipped {}"),
        (FAILED, "Failed to load {}"),
    ):
        exts = [ext for ext in extensions() if ext.status == status]
        if exts or status == LOADED:
            blocks.append(_block(title, exts))
    # A trailing blank line separates the report from what follows
    return '\n\n'.join(blocks) + '\n'


def _reset():
    """Allow load_extensions to run again (for tests)."""
    global _loaded
    _loaded = False

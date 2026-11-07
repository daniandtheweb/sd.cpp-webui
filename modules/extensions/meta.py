"""sd.cpp-webui - extensions - the metadata of an extension

An extension describes itself with top-level constants in its entry file
(the single .py file, or extension.py of a folder):

    NAME = 'Masker'
    VERSION = '1.0'
    AUTHOR = 'Someone'
    URL = 'https://example.com/masker'
    DESCRIPTION = 'What it does, in a sentence.'
    REQUIRES = ['rembg[cpu]']
    HOSTS = ['example.com']
    API_VERSION = 1
    PRIORITY = 0

REQUIRES and HOSTS can be a list, or a dict that says why each entry is
needed (shown when the extension is approved and in the Extensions tab):

    REQUIRES = {'rembg[cpu]': 'removes the background of a photo'}
    HOSTS = {'example.com': 'where the model is downloaded from'}

API_VERSION is mandatory: it is the version of the extensions API the
extension targets, and it only gets the functions of that version.

They are read with an AST parse, without importing the file: the
dependencies must be installed before the import, and the Extensions
tab can describe an extension that fails to load. Only literals count.
"""

import ast
import os
from dataclasses import dataclass, field


@dataclass
class Meta:
    name: str = ''
    version: str = ''
    author: str = ''
    url: str = ''
    description: str = ''
    requires: list[str] = field(default_factory=list)
    # The hosts it may contact through the api (see net.py)
    hosts: list[str] = field(default_factory=list)
    # entry -> why the extension needs it ('' when it doesn't say)
    require_reasons: dict[str, str] = field(default_factory=dict)
    host_reasons: dict[str, str] = field(default_factory=dict)
    api_version: int | None = None  # mandatory, see versions.py
    priority: int = 0
    # Problems found while reading the constants (shown in the log)
    problems: list[str] = field(default_factory=list)


_TEXT_FIELDS = {
    'NAME': 'name',
    'VERSION': 'version',
    'AUTHOR': 'author',
    'URL': 'url',
    'DESCRIPTION': 'description',
}
_INT_FIELDS = {
    'API_VERSION': 'api_version',
    'PRIORITY': 'priority',
}
_KNOWN = {*_TEXT_FIELDS, *_INT_FIELDS, 'REQUIRES', 'HOSTS'}


def read_meta(path: str, default_name: str) -> Meta:
    """Read the constants of the entry file at path."""
    meta = Meta(name=default_name)
    try:
        with open(path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError, ValueError) as e:
        meta.problems.append(f"cannot read {os.path.basename(path)}: {e}")
        return meta

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in _KNOWN:
                _apply(meta, target.id, value)
    return meta


def _apply(meta: Meta, const: str, node: ast.expr):
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        meta.problems.append(f"{const} must be a literal; ignored")
        return

    if const in _TEXT_FIELDS:
        if isinstance(value, str):
            setattr(meta, _TEXT_FIELDS[const], value.strip())
        else:
            meta.problems.append(f"{const} must be a string; ignored")
    elif const in _INT_FIELDS:
        if isinstance(value, int) and not isinstance(value, bool):
            setattr(meta, _INT_FIELDS[const], value)
        else:
            meta.problems.append(f"{const} must be an integer; ignored")
    elif const == 'HOSTS':
        _apply_hosts(meta, value)
    else:  # REQUIRES
        entries = _entries(meta, 'REQUIRES', value)
        if entries is not None:
            meta.requires = [k.strip() for k in entries if k.strip()]
            meta.require_reasons = {
                k.strip(): r for k, r in entries.items() if k.strip()
            }


def _entries(meta: Meta, const: str, value) -> dict[str, str] | None:
    """
    {entry: reason} of a string, a list of strings, or a dict of entry to
    reason (a reason may be left out of the first two: ''). None, with a
    problem noted, when it is none of those.
    """
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict):
        if all(isinstance(k, str) and isinstance(r, str)
               for k, r in value.items()):
            return {k: r.strip() for k, r in value.items()}
    elif (isinstance(value, (list, tuple))
          and all(isinstance(item, str) for item in value)):
        return dict.fromkeys(value, '')
    meta.problems.append(
        f"{const} must be a string, a list of strings or a dict of "
        "strings to their reasons; ignored"
    )
    return None


def _apply_hosts(meta: Meta, value):
    """HOSTS: names like 'example.com' or '*.example.com', no addresses."""
    from modules.extensions.net import valid_host_pattern

    entries = _entries(meta, 'HOSTS', value)
    if entries is None:
        return
    reasons = {k.strip().lower(): r for k, r in entries.items()}
    hosts = list(reasons)
    bad = [h for h in hosts if not valid_host_pattern(h)]
    if bad:
        meta.problems.append(
            f"HOSTS has invalid entries {bad!r}: use names like "
            "'example.com' or '*.example.com'; HOSTS ignored"
        )
        return
    meta.hosts = hosts
    meta.host_reasons = reasons

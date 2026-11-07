"""sd.cpp-webui - extensions - asking before running new code

An extension is Python code that runs with all your permissions, so the
app asks in the terminal before it runs one it has not seen: what it says
it does (network hosts, output folder, packages), then y/N. The answer is
tied to a checksum of the extension's files:

- approved: remembered. It loads without asking again until its files
  change (an update), when it asks again, saying it changed.
- rejected: not loaded, and nothing is remembered: the files stay where
  they are and the app asks again the next time it starts.
- no terminal: nobody can answer, so it counts as rejected.
- removed: at every start the approvals of extensions that are not in
  the folder any more are forgotten, so a copy put back later asks again.

The hosts and the folder shown are what the extension declares and what
the app gives it. This is not a sandbox: it can't tell you what the code
really does, only that it is the code you approved.
"""

import hashlib
import json
import os
import sys

from modules.extensions.host import Extension

STORE_NAME = 'trusted.json'


def store_path() -> str:
    """user_data/extensions/trusted.json, next to the config file."""
    from modules.extensions.host import _config
    base = os.path.dirname(os.path.abspath(_config().config_path))
    return os.path.join(base, 'extensions', STORE_NAME)


def interactive() -> bool:
    """Whether there is a terminal to ask in."""
    return bool(sys.stdin and sys.stdin.isatty())


def _files(entry: str, is_folder: bool) -> list[tuple[str, str]]:
    """(name, path) of every file that counts, in a fixed order."""
    if not is_folder:
        return [(os.path.basename(entry), entry)]
    root = os.path.dirname(entry)
    files = []
    for folder, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != '__pycache__')
        for name in sorted(names):
            if not name.endswith('.pyc'):
                full = os.path.join(folder, name)
                rel = os.path.relpath(full, root).replace(os.sep, '/')
                files.append((rel, full))
    return files


def checksum(entry: str, is_folder: bool) -> str:
    """
    SHA256 of the extension's files: the single file, or every file of
    its folder (names and contents, no caches).
    """
    digest = hashlib.sha256()
    for rel, full in _files(entry, is_folder):
        digest.update(rel.encode() + b'\0')
        try:
            with open(full, 'rb') as f:
                for chunk in iter(lambda: f.read(1 << 20), b''):
                    digest.update(chunk)
        except OSError:
            digest.update(b'<unreadable>')
        digest.update(b'\0')
    return digest.hexdigest()


def _read(path: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _remember(path: str, ext_id: str, digest: str):
    data = _read(path)
    data[ext_id] = digest
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def forget_missing(installed) -> list[str]:
    """
    Remove the approvals of extensions that are not installed any more
    (ids not in installed), so a copy put back later asks again. Returns
    the ids removed.
    """
    path = store_path()
    data = _read(path)
    gone = sorted(set(data) - set(installed))
    if not gone:
        return []
    for ext_id in gone:
        del data[ext_id]
    try:
        tmp = path + '.part'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError as e:
        print(f"Cannot remove the approval of {', '.join(gone)}: {e}")
        return []
    print(f"Removed the approval of extensions not installed: "
          f"{', '.join(gone)}")
    return gone


def _entry(name: str, reasons: dict) -> str:
    """A host or package with the reason the extension gives for it."""
    return f'    - {name}: {reasons.get(name) or "(no reason given)"}'


def describe(ext: Extension, digest: str, changed: bool) -> str:
    """What the dialog shows about an extension."""
    meta = ext.meta
    name = f"{ext.name} {meta.version}".strip()
    where = (os.path.dirname(ext.path) if os.path.basename(ext.path)
             == 'extension.py' else ext.path)
    lines = [
        '',
        '=' * 70,
        ("Updated extension: " if changed else "New extension: ") + name,
        '=' * 70,
    ]
    for label, value in (
        ('Author', meta.author), ('URL', meta.url),
        ('About', meta.description), ('Files', where),
        ('SHA256', digest),
    ):
        if value:
            lines.append(f'  {label + ":":<9}{value}')
    lines.append('')
    if meta.hosts:
        lines.append("  Network (declared by the extension, not enforced):")
        lines += [_entry(host, meta.host_reasons) for host in meta.hosts]
        lines.append(
            "  Files it downloads go to a folder the app picks: a model "
            f"folder or user_data/extensions/{ext.id}/."
        )
    else:
        lines.append("  Network: none declared.")
    lines.append(
        f"  If it saves any output, only here: "
        f"{ext.output_dir}"
    )
    if meta.requires:
        lines.append("  Packages it will ask to install:")
        lines += [_entry(req, meta.require_reasons)
                  for req in meta.requires]
    lines += [
        '',
        "  It runs as Python code with all your permissions, with no "
        "sandbox: it can do more than it declares. Only accept extensions "
        "you trust.",
        '',
    ]
    return '\n'.join(lines)


def ask(text: str) -> bool:
    """Show text and ask y/N in the terminal."""
    print(text)
    try:
        answer = input("Load this extension? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ('y', 'yes')


def approve(ext: Extension, is_folder: bool) -> str | None:
    """
    None when ext may load: it was approved before and is unchanged, or
    the user approves it now (and it is remembered). Otherwise why not.
    """
    digest = checksum(ext.path, is_folder)
    path = store_path()
    known = _read(path).get(ext.id)
    if known == digest:
        return None
    if not interactive():
        return (
            "not approved: a new or changed extension needs a yes in the "
            "terminal, and there is none"
        )
    if not ask(describe(ext, digest, changed=known is not None)):
        return "not approved; it will ask again on the next start"
    try:
        _remember(path, ext.id, digest)
    except OSError as e:
        print(f"{ext.id}: cannot remember the approval ({e}); "
              "it will ask again next time")
    return None

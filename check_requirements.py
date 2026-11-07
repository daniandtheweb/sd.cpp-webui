#!/usr/bin/env python3
"""sd.cpp-webui - check that the requirements are installed.

Used by the launch scripts, which install (visibly) only when this
fails. It only reads what is installed and prints one line per
requirement; it never installs anything. Standard library
only, so it runs in a fresh virtual environment before anything is
installed.

Usage: python check_requirements.py [requirements.txt]
Exit code: 0 when every requirement is met, 1 when something is missing
or has the wrong version.
"""

import importlib.metadata
import os
import re
import sys

_LINE = re.compile(
    r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;#]*)'
)
_SPEC = re.compile(r'\s*(==|>=|<=|!=|~=|>|<)\s*([A-Za-z0-9.*+!_-]+)\s*')


def _version_key(text: str) -> tuple:
    """Numeric tuple of a version; '1.2.3rc1' compares as (1, 2, 3)."""
    parts = []
    for piece in re.split(r'[.+-]', text):
        match = re.match(r'\d+', piece)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def _meets(installed: str, op: str, wanted: str) -> bool:
    if op == '==' and wanted.endswith('.*'):
        prefix = _version_key(wanted[:-2])
        return _version_key(installed)[:len(prefix)] == prefix
    have, want = _version_key(installed), _version_key(wanted)
    size = max(len(have), len(want))
    have += (0,) * (size - len(have))
    want += (0,) * (size - len(want))
    if op == '~=':
        base = _version_key(wanted)
        keep = max(len(base) - 1, 1)
        return have >= want and have[:keep] == want[:keep]
    return {
        '==': have == want, '!=': have != want,
        '>=': have >= want, '<=': have <= want,
        '>': have > want, '<': have < want,
    }[op]


def parse(path: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """(name, [(operator, version), ...]) of each requirement line."""
    found = []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(('#', '-')):
                continue
            match = _LINE.match(line)
            if not match:
                continue
            specs = _SPEC.findall(match.group(2))
            found.append((match.group(1), specs))
    return found


def check(path: str) -> list[tuple[bool, str]]:
    """(ok, message) for each requirement in the file."""
    results = []
    for name, specs in parse(path):
        wanted = ','.join(f'{op}{ver}' for op, ver in specs)
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            results.append((False, f'{name}{wanted}: not installed'))
            continue
        bad = [f'{op}{ver}' for op, ver in specs
               if not _meets(installed, op, ver)]
        if bad:
            results.append((
                False,
                f'{name}: {installed} is installed, needs {",".join(bad)}'
            ))
        else:
            results.append((True, f'{name} {installed}'))
    return results


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'requirements.txt'
    )
    try:
        results = check(path)
    except OSError as e:
        print(f'Cannot read {path}: {e}')
        return 1
    for ok, message in results:
        print(f'  [{"ok" if ok else "!!"}] {message}')
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))

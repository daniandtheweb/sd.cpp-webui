"""sd.cpp-webui - extensions - the dependencies of an extension

An extension's REQUIRES are installed when they are missing, into the
interpreter that runs the app -- after asking in the terminal, never
unconfirmed. That keeps the app's own requirements minimal: every
extension brings what it needs.

Entries are plain requirement strings with a version ('rembg[cpu]>=2.0,<3',
'pillow>=10'): a package without one would install whatever is newest,
which can break the extension later. Pip options, paths and direct URLs
are rejected, so a REQUIRES entry can only ever name a package from the
configured index.
"""

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


# What makes a version "given": a pin, or a minimum and/or a maximum
# ('!=' alone excludes a version and still allows anything else)
VERSION_OPERATORS = frozenset({'==', '===', '~=', '>=', '>', '<=', '<'})


def invalid_message(bad: list[str]) -> str:
    """Why entries were rejected, for the user."""
    return (
        'invalid REQUIRES entr' + ('y' if len(bad) == 1 else 'ies')
        + f': {", ".join(bad)} (every package needs a version, like '
        '"pkg>=1.0", "pkg>=1.0,<2" or "pkg==1.2"; no URLs or options)'
    )


def parse_requires(names: list[str]) -> tuple[list[Requirement], list[str]]:
    """
    Split names into valid requirements and rejected entries: invalid
    ones, direct URLs and packages without a version.
    """
    ok, bad = [], []
    for name in names:
        try:
            req = Requirement(name)
        except InvalidRequirement:
            bad.append(name)
            continue
        if req.url:  # 'pkg @ https://...' / 'pkg @ file:///...'
            bad.append(name)
            continue
        if not any(s.operator in VERSION_OPERATORS for s in req.specifier):
            bad.append(name)
            continue
        ok.append(req)
    return ok, bad


def _valid(text: str) -> bool:
    try:
        Requirement(text)
    except InvalidRequirement:
        return False
    return True


def is_satisfied(req: Requirement, _seen: frozenset = frozenset()) -> bool:
    """
    Whether req is available now: its distribution is installed (so pip
    names like 'pillow' work, not only import names) at an accepted
    version, and so are the dependencies its extras add -- 'rembg[cpu]'
    is not satisfied by a plain rembg, which has no onnxruntime.
    """
    if req.marker is not None and not req.marker.evaluate({'extra': ''}):
        return True  # does not apply to this platform

    try:
        dist = importlib.metadata.distribution(req.name)
    except importlib.metadata.PackageNotFoundError:
        # Not a distribution name; accept an importable module.
        return not req.specifier and not req.extras and _importable(req.name)

    if req.specifier and not req.specifier.contains(
        dist.version, prereleases=True
    ):
        return False

    key = canonicalize_name(req.name)
    if key in _seen:
        return True
    seen = _seen | {key}
    for extra in req.extras:
        for line in dist.requires or ():
            try:
                sub = Requirement(line)
            except InvalidRequirement:
                continue
            # Only what the extra adds; the base dependencies are pip's job
            if sub.marker is None or 'extra' not in str(sub.marker):
                continue
            if not sub.marker.evaluate({'extra': extra}):
                continue
            if not is_satisfied(Requirement(sub.name + str(sub.specifier)),
                                seen):
                return False
    return True


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing(reqs: list[Requirement]) -> list[Requirement]:
    return [req for req in reqs if not is_satisfied(req)]


def _installer(args: list[str]) -> list[str] | None:
    """
    The command that installs args into the running interpreter: pip
    when it is there, uv for environments made by uv (they have no pip),
    else pip bootstrapped with ensurepip.
    """
    if importlib.util.find_spec('pip') is not None:
        return [sys.executable, '-m', 'pip', 'install', *args]
    uv = shutil.which('uv')
    if uv:
        return [uv, 'pip', 'install', '--python', sys.executable, *args]
    try:
        subprocess.run(
            [sys.executable, '-m', 'ensurepip', '--upgrade'],
            check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    importlib.invalidate_caches()
    return [sys.executable, '-m', 'pip', 'install', *args]


def install(reqs: list[Requirement]) -> str | None:
    """Install reqs; return None on success, else the reason it failed."""
    args = [str(req) for req in reqs]
    cmd = _installer(args)
    if cmd is None:
        return (
            "no installer: this Python has neither pip nor uv "
            f'(install by hand: {" ".join(args)})'
        )
    print(f"Installing: {' '.join(cmd)}")
    try:
        # The output streams to the terminal: big packages take minutes
        proc = subprocess.run(cmd, check=False)
    except OSError as e:
        return f"installer failed to start: {e}"
    if proc.returncode != 0:
        return f"installer exited with code {proc.returncode}"
    importlib.invalidate_caches()
    return None


RISK_WARNING = (
    "\nWARNING: extensions and the packages they require are code that "
    "runs with all your permissions, with no sandbox. Installing a "
    "package can run its code right away. Only continue if you trust "
    "this extension and where it came from."
)


def ask_install(
    ext_name: str, reqs: list[Requirement], reasons: dict | None = None
) -> bool:
    """
    Ask in the terminal whether to install reqs for an extension;
    reasons (requirement as the extension wrote it -> why) are shown.
    Without a terminal there is nobody to ask, so the answer is no.
    """
    if not sys.stdin or not sys.stdin.isatty():
        return False
    print(RISK_WARNING)
    why = {canonicalize_name(Requirement(k).name): r
           for k, r in (reasons or {}).items() if r and _valid(k)}
    for req in reqs:
        if canonicalize_name(req.name) in why:
            print(f"  {req.name}: {why[canonicalize_name(req.name)]}")
    try:
        answer = input(
            f"Extension '{ext_name}' requires missing package(s): "
            f"{', '.join(str(req) for req in reqs)}. Install now? [y/N] "
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in ('y', 'yes')


def ensure(names: list[str], confirm=None) -> str | None:
    """
    Make sure the requirements in names are installed. What is missing
    is installed only if confirm(missing requirements) says yes (no
    confirm means no). Return None when all are available, else why not.
    """
    reqs, bad = parse_requires(names)
    if bad:
        return invalid_message(bad)
    todo = missing(reqs)
    if not todo:
        return None
    by_hand = " ".join(str(req) for req in todo)
    if confirm is None or not confirm(todo):
        return (
            f"missing package(s) not installed ({by_hand}); "
            "install them with your Python package manager "
            "and restart, or answer y when asked"
        )
    error = install(todo)
    if error:
        return error
    still = missing(todo)
    if still:
        return (
            "still missing after install: "
            + ", ".join(str(req) for req in still)
        )
    return None

"""sd.cpp-webui - Option registry - what each option reaches

    python -m modules.options.report            every option, by status
    python -m modules.options.report scheduler  where one option goes

Statuses:
    both         output on the CLI and on the server
    cli only     nothing for the server: the reason (Option.no_server) is
                 shown, or that the server has no such tab (video,
                 upscale, convert)
    server only  only the server gets it (the CLI gets it another way)
    toggle       passes nothing itself, it only switches other options
                 on or off (a checkbox); the options it gates are shown
    ui only      not passed to sd.cpp: a config setting, ...

This module must not import application modules.
"""

import re
import sys
from typing import Dict, List

from modules.options import registry
from modules.options.spec import (
    DEFAULT_SCOPES, IMAGE_KINDS, Option, expand_scopes
)

SERVER_KINDS = set(IMAGE_KINDS)


def cli_kinds(option: Option) -> List[str]:
    """The sd-cli runner kinds the option is output for."""
    kinds = set()
    for out in option.cli:
        kinds |= expand_scopes(out.scopes or DEFAULT_SCOPES['cli'])
    return sorted(kinds)


def launch_flags(option: Option) -> List[str]:
    """The sd-server startup flags of the option."""
    return sorted({out.flag for out in option.launch})


def payload_fields(option: Option) -> List[str]:
    """
    The request fields of the option, those it sets as the toggle of a
    group (the marker, the off state) included.
    """
    fields = []
    for out in option.payload:
        fields.extend((out.path,) if isinstance(out.path, str) else out.path)
    for group in registry.GROUPS.values():
        if group.toggle == option.name:
            if group.marker:
                fields.append(group.marker)
            fields.extend(group.off)
    return sorted(set(fields))


def _condition_names(option: Option) -> List[str]:
    """Names of the settings the conditions of an option look at."""
    conditions = list(option.when)
    for out in (*option.cli, *option.launch, *option.payload):
        conditions.extend(out.when)
    group = registry.GROUPS.get(option.group) if option.group else None
    if group:
        conditions.append(group.toggle)
    names = []
    for cond in conditions:
        if cond.startswith('@'):
            continue
        name = re.split(r'[=!]=', cond.lstrip('!?'))[0]
        names.append(name.split(':')[-1])
    return names


def gates() -> Dict[str, List[str]]:
    """Option name -> the options its value switches on or off."""
    gated: Dict[str, List[str]] = {}
    for option in registry.OPTIONS:
        for name in set(_condition_names(option)):
            if name != option.name:
                gated.setdefault(name, []).append(option.name)
    return gated


def _group_has_cli(option: Option) -> bool:
    """True for the toggle of a group whose options are in the CLI."""
    return any(
        o.cli for o in registry.OPTIONS
        if o.group and registry.GROUPS[o.group].toggle == option.name
    )


def status(option: Option) -> str:
    cli = bool(option.cli)
    server = bool(option.launch or payload_fields(option))
    if cli and server:
        return 'both'
    if cli:
        return 'cli only'
    if server:
        # A group's toggle also switches its options in the CLI
        return 'both' if _group_has_cli(option) else 'server only'
    if option.name in gates():
        return 'toggle'
    return 'ui only'


def needs_a_reason(option: Option) -> bool:
    """
    True for an option of the image tabs (the only ones the server has)
    that has CLI output but nothing for the server: it must say why.
    """
    return (status(option) == 'cli only'
            and bool(set(cli_kinds(option)) & SERVER_KINDS))


def _note(option: Option) -> str:
    if status(option) == 'toggle':
        return 'switches: ' + ', '.join(gates()[option.name])
    if option.no_server:
        return option.no_server
    if status(option) == 'cli only':
        kinds = ', '.join(cli_kinds(option))
        return f"the server has no {kinds} tab"
    return ''


def _flags(option: Option) -> str:
    return ' '.join(sorted({o.flag for o in option.cli}))


def describe(option: Option) -> str:
    """Multi-line description of where one option goes."""
    lines = [f"{option.name}  ({status(option)})"]
    if not option.virtual:
        lines.append(f"  ui key      {option.ui_key}")
    if option.cli:
        lines.append(f"  sd-cli      {_flags(option)}   "
                     f"[{', '.join(cli_kinds(option))}]")
    if option.launch:
        lines.append(f"  sd-server   {' '.join(launch_flags(option))}"
                     "   [startup command]")
    if payload_fields(option):
        lines.append(f"  request     {', '.join(payload_fields(option))}")
    if _note(option):
        lines.append(f"  note        {_note(option)}")
    return '\n'.join(lines)


def report() -> str:
    groups = {'both': [], 'cli only': [], 'server only': [], 'toggle': [],
              'ui only': []}
    for option in registry.OPTIONS:
        groups[status(option)].append(option)

    out = []
    width = max(len(o.name) for o in registry.OPTIONS)
    titles = {
        'both': 'BOTH: sd-cli and sd-server',
        'cli only': 'CLI ONLY: nothing for the server',
        'server only': 'SERVER ONLY',
        'toggle': 'TOGGLES: only switch other options',
        'ui only': 'UI ONLY: not passed to sd.cpp',
    }
    for key, title in titles.items():
        options = groups[key]
        out.append(f"{title} ({len(options)})")
        out.append('-' * len(out[-1]))
        for o in options:
            if key == 'both':
                where = []
                if o.launch:
                    where.append('startup ' + ' '.join(launch_flags(o)))
                if payload_fields(o):
                    where.append('request ' + ', '.join(payload_fields(o)))
                out.append(f"  {o.name:<{width}}  {_flags(o):<30} "
                           f"{' | '.join(where)}")
            elif key == 'cli only':
                out.append(f"  {o.name:<{width}}  {_flags(o):<30} "
                           f"{_note(o)}")
            elif key == 'server only':
                where = []
                if o.launch:
                    where.append('startup ' + ' '.join(launch_flags(o)))
                if payload_fields(o):
                    where.append('request ' + ', '.join(payload_fields(o)))
                out.append(f"  {o.name:<{width}}  {' | '.join(where)}")
            elif key == 'toggle':
                out.append(f"  {o.name:<{width}}  {_note(o)[10:]}")
            else:
                out.append(f"  {o.name}")
        out.append('')
    return '\n'.join(out)


def main(argv: List[str]) -> int:
    if not argv:
        print(report())
        return 0
    by_name = {o.name: o for o in registry.OPTIONS}
    for name in argv:
        name = name.removeprefix('in_')
        if name not in by_name:
            print(f"{name}: not registered", file=sys.stderr)
            return 1
        print(describe(by_name[name]))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

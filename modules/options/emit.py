"""sd.cpp-webui - Option registry - emission

Turns the UI parameters into what sd.cpp receives, following the
registry (registry.OPTIONS):

    cli(ctx)      tokens of an sd-cli command     (ctx.kind: txt2img, ...)
    launch(ctx)   tokens of the sd-server startup command (kind 'server',
                  or 'video' for the video tab)
    payload(ctx)  the per-request JSON of sd-server (kind: txt2img.., video)

The runners only add what the registry can't know (executable, output
paths, images, LoRAs, ...), passing those they compute in ctx.values and
ctx.flags.

Conditions (Option.when / Out.when), all of them must hold:
    name          the UI setting 'in_<name>' is truthy ('f:name': 'f_<name>')
    !name         it's falsy
    name==value   it equals the value (a number or a string)
    name!=value   it doesn't
    ?name         the tab has the setting at all
    @flag         the runner set ctx.flags['flag']

This module must not import application modules.
"""

from typing import Any, Callable, Dict, List, Optional

from modules.options import convert as converters
from modules.options import registry
from modules.options.spec import (
    DEFAULT_SCOPES, Group, Option, Out, expand_scopes
)


class Context:
    """
    What the emission needs to know about one run.

    params:  the UI parameters ('in_*', plus the resolved paths 'f_*').
    kind:    txt2img, img2img, imgedit, video, upscale, convert (CLI),
             server, video (launch command); the payload uses the image kinds
             and video.
    rel:     makes a path relative to the executable.
    cfg:     reads the config (cfg:<key> sources).
    values:  values the runner computes (ctx:<name> sources).
    flags:   conditions the runner computes (@<name> conditions).
    """

    def __init__(self, params: Dict[str, Any], kind: str,
                 rel: Callable[[Any], Any] = lambda p: p,
                 cfg: Callable[[str], Any] = lambda key: None,
                 values: Optional[Dict[str, Any]] = None,
                 flags: Optional[Dict[str, bool]] = None):
        self.params = params
        self.kind = kind
        self.rel = rel
        self.cfg = cfg
        self.values = values or {}
        self.flags = flags or {}


def _same(a: Any, b: Any) -> bool:
    """Equality that keeps bools apart from numbers (False != 0)."""
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _key(name: str) -> str:
    """'f:name' -> 'f_name' (a resolved model path); 'name' -> 'in_name'."""
    if name.startswith('f:'):
        return f'f_{name[2:]}'
    return f'in_{name}'


def _literal(text: str) -> Any:
    try:
        return int(text)
    except ValueError:
        return text


def check(condition: str, ctx: Context) -> bool:
    """Evaluates one condition (see the module doc)."""
    if condition.startswith('@'):
        return bool(ctx.flags.get(condition[1:]))
    if condition.startswith('!'):
        return not check(condition[1:], ctx)
    if condition.startswith('?'):
        return _key(condition[1:]) in ctx.params
    for op in ('==', '!='):
        name, sep, literal = condition.partition(op)
        if sep:
            equal = ctx.params.get(_key(name)) == _literal(literal)
            return equal if op == '==' else not equal
    return bool(ctx.params.get(_key(condition)))


def _group(option: Option) -> Optional[Group]:
    return registry.GROUPS[option.group] if option.group else None


def _read(source, option: Option, ctx: Context) -> Any:
    """Reads the raw value from its source."""
    if isinstance(source, tuple):
        for item in source:
            value = _read(item, option, ctx)
            if value:
                return value
        return None
    if source is None:
        if option.virtual:
            return None
        return ctx.params.get(option.ui_key)
    kind, sep, rest = source.partition(':')
    if sep and kind == 'ctx':
        return ctx.values.get(rest)
    if sep and kind == 'cfg':
        return ctx.cfg(rest)
    return ctx.params.get(_key(source))


def _omitted(value: Any, omit) -> bool:
    return any(_same(value, o) for o in omit)


def resolve(option: Option, out: Out, ctx: Context) -> Any:
    """
    The value an output sends, or None to leave it out:
    read -> fallback -> omit -> convert -> omit -> relative.
    """
    omit = option.omit if out.omit is None else out.omit
    source = out.source if out.source is not None else option.source
    raw = _read(source, option, ctx)
    if isinstance(raw, str) and option.strip:
        raw = raw.strip()
    if out.fallback is not None and raw in (None, ''):
        raw = out.fallback

    if _omitted(raw, omit):
        return None
    value = converters.convert(out.convert, raw, ctx) if out.convert else raw
    if value is None or _omitted(value, omit):
        return None
    if out.relative:
        if isinstance(value, list):
            return [ctx.rel(v) for v in value]
        return ctx.rel(str(value))
    return value


def _active(option: Option, out: Out, ctx: Context, target: str) -> bool:
    """Scope and conditions (group toggle included) of an output."""
    scopes = expand_scopes(out.scopes or DEFAULT_SCOPES[target])
    if ctx.kind not in scopes:
        return False
    group = _group(option)
    if group:
        if (target == 'payload' and group.requires_key
                and _key(group.toggle) not in ctx.params):
            return False
        if out.gated and not check(group.toggle, ctx):
            return False
    return all(check(c, ctx) for c in option.when + out.when)


def _command(target: str, ctx: Context) -> List[str]:
    tokens: List[str] = []
    for option in registry.OPTIONS:
        for out in getattr(option, target):
            if not out.flag or not _active(option, out, ctx, target):
                continue
            value = resolve(option, out, ctx)
            if value is None or value is False:
                continue
            if out.kind == 'flag':
                if value:
                    tokens.append(out.flag)
            elif out.kind == 'repeat':
                for item in value:
                    tokens.extend([out.flag, str(item)])
            else:
                tokens.extend([out.flag, str(value)])
    return tokens


def cli(ctx: Context) -> List[str]:
    """Tokens of the sd-cli command (after the executable and -M mode)."""
    return _command('cli', ctx)


def cli_value(name: str, tokens: List[str]) -> Optional[str]:
    """The value of an option in the tokens of a command, None if absent."""
    option = next(o for o in registry.OPTIONS if o.name == name)
    for out in option.cli:
        if out.flag in tokens[:-1]:
            return tokens[tokens.index(out.flag) + 1]
    return None


def launch(ctx: Context) -> List[str]:
    """Tokens of the sd-server startup command."""
    return _command('launch', ctx)


def _set(payload: Dict[str, Any], path: str, value: Any):
    *parents, leaf = path.split('.')
    target = payload
    for key in parents:
        target = target.setdefault(key, {})
    target[leaf] = value


def _set_fields(ctx: Context, result: Dict[str, Any]):
    """The payload fields of the options."""
    for option in registry.OPTIONS:
        for out in option.payload:
            if not out.path or not _active(option, out, ctx, 'payload'):
                continue
            value = resolve(option, out, ctx)
            if value is None:
                continue
            if out.kind == 'flag':
                if not value:
                    continue
                value = True
            for path in ((out.path,) if isinstance(out.path, str)
                         else out.path):
                _set(result, path, value)


def _set_group_states(ctx: Context, result: Dict[str, Any]):
    """What each group says about its toggle: the marker, the off fields."""
    for group in registry.GROUPS.values():
        kinds = expand_scopes(group.scopes or DEFAULT_SCOPES['payload'])
        if ctx.kind not in kinds or not (group.marker or group.off):
            continue
        if group.requires_key and _key(group.toggle) not in ctx.params:
            continue
        on = check(group.toggle, ctx)
        if group.marker:
            _set(result, group.marker, on)
        if not on:
            for path, value in group.off.items():
                _set(result, path, value)


def payload(ctx: Context, base: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
    """Sets the registered fields in a request payload (and returns it)."""
    result = base if base is not None else {}
    _set_fields(ctx, result)
    _set_group_states(ctx, result)
    return result

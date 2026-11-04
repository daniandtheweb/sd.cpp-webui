"""sd.cpp-webui - Option registry - data model

An Option describes one setting and everything the application does with
it. Per target it can have one or more outputs (Out):

- cli:     the sd-cli command (txt2img, img2img, imgedit, video, upscale,
           convert)
- launch:  the sd-server startup command
- payload: the JSON sent to sd-server with every request

so the CLI and server runners only run the registry (modules/options/
emit.py) instead of each writing the flag/field by hand.

This module must not import application modules: config.py imports it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

# No config default: the setting is UI-only (or only exists once saved).
NO_DEFAULT = object()

# What a command is generated for. Targets use them as 'scopes'.
IMAGE_KINDS = ('txt2img', 'img2img', 'imgedit')
GEN_KINDS = IMAGE_KINDS + ('video',)
ALL_KINDS = GEN_KINDS + ('upscale',)

SCOPE_ALIASES = {
    'image': IMAGE_KINDS,      # txt2img, img2img, imgedit
    'gen': GEN_KINDS,          # image + video
    'all': ALL_KINDS,          # gen + upscale (not convert)
}

# Default scopes of each target
DEFAULT_SCOPES = {
    'cli': ('all',),
    'launch': ('server',),
    'payload': ('image',),
}


def expand_scopes(scopes: Tuple[str, ...]) -> frozenset:
    """Replaces the aliases ('image', 'gen', 'all') by the runner kinds."""
    kinds = set()
    for name in scopes:
        kinds.update(SCOPE_ALIASES.get(name, (name,)))
    return frozenset(kinds)


@dataclass(frozen=True)
class Out:
    """
    How an option is output on one target.

    flag:     CLI flag ('--steps'), for the cli and launch targets.
    path:     dotted payload path ('sample_params.sample_steps'); a tuple
              sets several fields. For the payload target.
    kind:     'value'  flag followed by the value (payload: the value),
              'flag'   bare flag (payload: true), output when truthy,
              'repeat' flag repeated for each item of a list.
    scopes:   runner kinds it applies to: txt2img, img2img, imgedit,
              video, upscale, convert, server (launch); or an alias
              (image, gen, all). Empty: the target's default.
    when:     extra conditions, all must hold (see emit.py): 'name',
              '!name', 'name==value', 'name!=value', '?name' (the tab
              has the setting), '@flag' (set by the runner).
    omit:     values that mean "leave it to sd.cpp" (not output). None:
              the option's own.
    convert:  named converter (convert.py) applied to the value.
    fallback: used when the value is missing or empty.
    source:   where the value comes from, instead of the option's own
              'in_<name>': 'name' (another UI setting), 'f:name' (resolved
              model path, 'f_<name>'), 'cfg:key' (config), 'ctx:name'
              (computed by the runner), or a tuple: the first truthy one
              wins.
    relative: make the path(s) relative to the executable.
    gated:    False: output even when the option's group is off.
    """
    flag: Optional[str] = None
    path: Union[None, str, Tuple[str, ...]] = None
    kind: str = 'value'
    scopes: Tuple[str, ...] = ()
    when: Tuple[str, ...] = ()
    omit: Optional[Tuple[Any, ...]] = None
    convert: Optional[str] = None
    fallback: Any = None
    source: Union[None, str, Tuple[str, ...]] = None
    relative: bool = False
    gated: bool = True


@dataclass(frozen=True)
class Group:
    """
    A section with an on/off toggle (HiRes Fix, Cache, VAE tiling, ...).

    Its options are only output while the toggle is on (unless the Out is
    not 'gated'). For the payload, the group also says what an off or
    missing toggle looks like.

    toggle:       name of the checkbox ('hires_bool').
    requires_key: payload only: leave the group out when the tab has no
                  such checkbox.
    marker:       payload path set to the toggle state
                  ('hires.enabled': true/false).
    off:          payload fields set while the toggle is off
                  ({'cache_mode': 'disabled'}).
    """
    name: str
    toggle: str
    requires_key: bool = False
    marker: Optional[str] = None
    off: Dict[str, Any] = field(default_factory=dict)


def _outs(value, target: str) -> Tuple[Out, ...]:
    """Normalizes cli=/launch=/payload= to a tuple of Out."""
    if not value:
        return ()
    if isinstance(value, (str, Out)):
        value = (value,)
    outs = []
    for item in value:
        if isinstance(item, str):
            item = Out(path=item) if target == 'payload' else Out(flag=item)
        outs.append(item)
    return tuple(outs)


@dataclass(frozen=True)
class Option:
    """
    One setting.

    name:     base name; the UI key is '<prefix>_<name>' ('in_<name>'),
              the config key 'def_<name>'.
    default:  the config default (NO_DEFAULT: none).
    cli/launch/payload: what to output (str, Out or a list of them; a
              str is a flag for cli/launch and a path for payload).
    omit:     values meaning "leave it to sd.cpp's default" (not output):
              compared with bool/number kept apart (False != 0).
    when:     conditions for every output (see Out.when).
    group:    name of its Group.
    strip:    strip the string before comparing/using it.
    virtual:  not a UI setting: it only outputs something (a value from
              the config or computed from other settings). No config key.
              One assembled by a converter from other settings has no
              value of its own to compare: it says omit=(), and the
              converter returns None to leave it out.
    source:   see Out.source; default: the UI key.
    prefix:   UI key prefix, 'in' by default; '' for the environment
              variables, whose UI key is the name itself.
    mapped:   how the Settings tab finds the setting's config key:
              True   the tab reuses a block of another tab (quantization,
                     performance, extras, ...) and saves its widgets,
                     keyed 'in_<name>', through the generated key map
                     'in_<name>' -> 'def_<name>'
                     (registry.config_key_map()).
              False  the tab registers the widget itself under
                     'def_<name>' (generation settings, output, ...), or
                     the setting has no config key (UI-only).
              Default: True when there is a default value, False without
              one. So only the settings that have a default but are
              registered by hand in settings_tab.py say mapped=False.
    path_dir: config key of the folder the model name is relative to;
              makes the runners resolve 'f_<name>' from it.
    default_key: the config key a default model is saved under, when it
              isn't 'def_<name>' ('def_ckpt' for 'ckpt_model'). Only for
              model files (see path_dir).
    no_server: why an option of the image tabs has no sd-server output
              (no launch= nor payload=): the reason, in a few words. The
              tests ask for it so that nothing is left out by accident;
              `python -m modules.options.report` lists them.
    """
    name: str
    default: Any = NO_DEFAULT
    cli: Any = ()
    launch: Any = ()
    payload: Any = ()
    omit: Tuple[Any, ...] = (None, '')
    when: Tuple[str, ...] = ()
    group: Optional[str] = None
    strip: bool = False
    virtual: bool = False
    source: Union[None, str, Tuple[str, ...]] = None
    prefix: str = 'in'
    mapped: Optional[bool] = None
    path_dir: Optional[str] = None
    default_key: Optional[str] = None
    no_server: str = ''

    def __post_init__(self):
        for target in ('cli', 'launch', 'payload'):
            object.__setattr__(
                self, target, _outs(getattr(self, target), target)
            )
        if self.mapped is None:
            object.__setattr__(
                self, 'mapped',
                self.default is not NO_DEFAULT and not self.virtual
            )

    @property
    def ui_key(self) -> str:
        return f'{self.prefix}_{self.name}' if self.prefix else self.name

    @property
    def config_key(self) -> str:
        """Where the setting's default is saved in the config."""
        return self.default_key or f'def_{self.name}'

    @property
    def resolved_key(self) -> str:
        """Key of the resolved model path ('f_<name>')."""
        return f'f_{self.name}'

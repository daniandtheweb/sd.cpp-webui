"""Each registered output works on the runner kinds it targets."""
import pytest

from modules.options import emit, registry, report
from modules.options.emit import _key, _literal
from modules.options.registry import OPTIONS
from modules.options.spec import (
    DEFAULT_SCOPES, IMAGE_KINDS, expand_scopes
)

# A value that must come out of each converter
SAMPLES = {
    'int': 3, 'float': 1.5, 'text': 'x', 'not': True, 'seed': 7,
    'int_or_1': 2, 'strip_quotes': 'a=1', 'dynamic_policy': 'dynamic',
    'tile_square': 32, 'stem': 'a/b.pth', 'splitext0': 'a/b.pth',
    'sampler_api': 'euler a', 'scheduler_api': 'karras',
    'number_list': '1, 2', 'int_list': '1, 2', 'image_paths': ['a.png'],
    'backend_table': [['*', 'cuda0']], 'params_backend_table': [['*', 'cpu']],
    'split_modes': [['diffusion', 'row']], 'max_vram': [['*', '8']],
    'mode_spec': 'on',
}
# Converters that assemble their value from other settings
ASSEMBLED = {
    'model_args': {'in_chroma_use_t5_mask': True},
    'ad_args': {'in_ad_mask_blur': 9},
    'ref_image_args': {'in_increase_ref_index': True},
    'qwen_layers': {'in_qwen_layered_bool': True, 'in_qwen_image_layers': 2},
}
KINDS = {
    'cli': set(expand_scopes(DEFAULT_SCOPES['cli'])) | {'convert'},
    'launch': {'server'},
    'payload': set(IMAGE_KINDS),
}


def _setup(option, out, target):
    """Parameters under which the output must fire."""
    params, values, flags, cfg = {}, {}, {}, {}
    fixed = set()                       # keys a condition has set

    conditions = list(option.when + out.when)
    group = registry.GROUPS[option.group] if option.group else None
    if group and out.gated:
        conditions.append(group.toggle)
    elif group and group.requires_key and target == 'payload':
        # not gated by the toggle, but the tab must have the checkbox
        params[_key(group.toggle)] = False
    for cond in conditions:
        if cond.startswith('@'):
            flags[cond[1:]] = True
        elif cond.startswith('!') or '!=' in cond:
            continue                    # a missing setting satisfies these
        elif cond.startswith('?'):
            params[_key(cond[1:])] = True
        elif '==' in cond:
            name, literal = cond.split('==', 1)
            params[_key(name)] = _literal(literal)
            fixed.add(_key(name))
        else:
            params[_key(cond)] = True

    source = out.source if out.source is not None else option.source
    source = source[0] if isinstance(source, tuple) else source
    convert_name, _, arg = (out.convert or '').partition(':')
    omitted = option.omit if out.omit is None else out.omit

    if convert_name in ASSEMBLED:
        params.update(ASSEMBLED[convert_name])
        return params, values, flags, cfg
    if option.virtual and source is None:
        return params, values, flags, cfg   # a constant

    if convert_name in SAMPLES:
        sample = SAMPLES[convert_name]
    elif out.kind == 'repeat':
        sample = ['a', 'b']
    elif out.kind == 'flag':
        sample = True
    else:
        sample = next(v for v in ('x', 3, 1.5) if v not in omitted)

    if source and source.startswith('ctx:'):
        values[source[4:]] = sample if isinstance(sample, list) else 'V'
    elif source and source.startswith('cfg:'):
        cfg[source[4:]] = 'folder'
    elif out.convert in ('api_preview_mode', 'api_preview_interval',
                         'listen_ip', 'listen_port'):
        pass                            # always output
    else:
        key = _key(source) if source else option.ui_key
        if key not in fixed:
            params[key] = sample
        if convert_name == 'mode_spec':
            params[f'in_{arg}'] = 'on'
    return params, values, flags, cfg


def _has(payload, path):
    for key in path.split('.'):
        if not isinstance(payload, dict) or key not in payload:
            return False
        payload = payload[key]
    return True


def _cases():
    for option in OPTIONS:
        for target in ('cli', 'launch', 'payload'):
            for out in getattr(option, target):
                scopes = expand_scopes(out.scopes or DEFAULT_SCOPES[target])
                for kind in sorted(scopes & KINDS[target]):
                    yield pytest.param(
                        option, out, target, kind,
                        id=f"{option.name}-{target}-{kind}"
                           f"-{out.flag or out.path}")


@pytest.mark.parametrize('option, out, target, kind', _cases())
def test_every_output_fires_on_its_runner_kinds(option, out, target, kind):
    params, values, flags, cfg = _setup(option, out, target)
    context = emit.Context(params, kind, cfg=cfg.get, values=values,
                           flags=flags)
    if target == 'payload':
        result = emit.payload(context, {'sample_params': {}})
        paths = (out.path,) if isinstance(out.path, str) else out.path
        assert all(_has(result, p) for p in paths), result
    else:
        tokens = (emit.cli if target == 'cli' else emit.launch)(context)
        assert out.flag in tokens, tokens


# ---------------------------------------------------------------------------
# Which options the server doesn't get
# ---------------------------------------------------------------------------

def test_every_image_option_reaches_the_server_or_says_why():
    """
    An option of the image tabs with CLI output but nothing for the server
    must carry no_server='reason', so it's a decision and never an
    oversight. Add launch= / payload= (or the reason) to the new option.
    """
    missing = [o.name for o in OPTIONS
               if report.needs_a_reason(o) and not o.no_server]
    assert not missing, (
        f"no server output and no no_server= reason: {missing}")


def test_no_server_reasons_are_true():
    for o in OPTIONS:
        if o.no_server:
            assert o.cli, f"{o.name}: no_server= without a CLI output"
            assert not (o.launch or o.payload), \
                f"{o.name}: has a server output, drop no_server="
            assert report.needs_a_reason(o), \
                f"{o.name}: no_server= on an option the server never gets"


def test_report_runs_and_covers_every_option():
    text = report.report()
    for o in OPTIONS:
        assert o.name in text
    for o in OPTIONS:
        assert o.name in report.describe(o)


@pytest.mark.parametrize('name, expected', [
    ('scheduler', 'both'), ('hires_bool', 'both'), ('cache_bool', 'both'),
    ('preview_taesd', 'both'),
    ('preview_noisy', 'cli only'), ('frames', 'cli only'),
    ('ip', 'server only'), ('pprompt', 'server only'),
    ('max_vram_bool', 'toggle'), ('theme', 'ui only'),
])
def test_status(name, expected):
    option = next(o for o in OPTIONS if o.name == name)
    assert report.status(option) == expected

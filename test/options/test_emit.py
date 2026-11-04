"""Option registry engine (modules/options/emit.py) on small registries."""
import pytest

from modules.options import convert, emit, registry
from modules.options.spec import Group, Option, Out


def use(monkeypatch, *options, groups=()):
    """Replaces the registry by the given options and groups."""
    monkeypatch.setattr(registry, 'OPTIONS', options)
    monkeypatch.setattr(registry, 'GROUPS', {g.name: g for g in groups})


def ctx(params, kind='txt2img', **kw):
    return emit.Context(params, kind, **kw)


# ---------------------------------------------------------------------------
# Values: omit, fallback, convert
# ---------------------------------------------------------------------------

def test_value_flag(monkeypatch):
    use(monkeypatch, Option('steps', cli='--steps'))
    assert emit.cli(ctx({'in_steps': 20})) == ['--steps', '20']


@pytest.mark.parametrize('missing', [None, ''])
def test_none_and_empty_are_omitted(monkeypatch, missing):
    use(monkeypatch, Option('steps', cli='--steps'))
    assert emit.cli(ctx({'in_steps': missing})) == []
    assert emit.cli(ctx({})) == []


def test_omit_keeps_bools_apart_from_numbers(monkeypatch):
    use(monkeypatch, Option('x', cli='--x', omit=(None, 0)),
        Option('y', cli='--y', omit=(None, False)))
    assert emit.cli(ctx({'in_x': 0})) == []
    assert emit.cli(ctx({'in_x': 0.0})) == []   # gr.Number may return floats
    assert emit.cli(ctx({'in_y': 0})) == ['--y', '0']
    assert emit.cli(ctx({'in_y': False})) == []


def test_omit_sdcpp_default(monkeypatch):
    use(monkeypatch, Option('fit', cli='--auto-fit', omit=(None, '', 'on')))
    assert emit.cli(ctx({'in_fit': 'on'})) == []
    assert emit.cli(ctx({'in_fit': 'off'})) == ['--auto-fit', 'off']


def test_out_omit_overrides_the_options(monkeypatch):
    use(monkeypatch, Option(
        'q', omit=(None, '', 90), cli='--q',
        payload=Out(path='output_compression', omit=(None, ''))))
    assert emit.cli(ctx({'in_q': 90})) == []
    assert emit.payload(ctx({'in_q': 90})) == {'output_compression': 90}


def test_strip_option(monkeypatch):
    use(monkeypatch,
        Option('a', cli='--a', strip=True), Option('b', cli='--b'))
    assert emit.cli(ctx({'in_a': ' x ', 'in_b': ' y '})) == [
        '--a', 'x', '--b', ' y ']
    assert emit.cli(ctx({'in_a': '   '})) == []


def test_fallback_when_missing_or_empty(monkeypatch):
    use(monkeypatch, Option('w', payload=Out(path='width', convert='int',
                                             fallback=512)))
    assert emit.payload(ctx({})) == {'width': 512}
    assert emit.payload(ctx({'in_w': ''})) == {'width': 512}
    assert emit.payload(ctx({'in_w': 640.0})) == {'width': 640}


def test_convert_and_converter_returning_none(monkeypatch):
    use(monkeypatch, Option('w', cli=Out('-W', convert='int')))
    assert emit.cli(ctx({'in_w': 512.0})) == ['-W', '512']
    assert emit.cli(ctx({'in_w': 'oops'})) == []


def test_converter_argument(monkeypatch):
    use(monkeypatch, Option(
        'flash_attention', omit=(),
        cli=Out('--flash-attn', convert='mode_spec:flash_attention')))
    assert emit.cli(ctx({})) == []
    assert emit.cli(ctx({'in_flash_attention': 'on',
                         'in_flash_attention_vae': 'off'})) == [
        '--flash-attn', 'on,vae=off']


def test_relative_paths(monkeypatch):
    use(monkeypatch, Option('m', cli=Out('--model', relative=True)),
        Option('imgs', cli=Out('--ref-image', kind='repeat', relative=True)))
    rel = lambda p: f"rel/{p}"  # noqa: E731
    assert emit.cli(ctx({'in_m': 'a.gguf'}, rel=rel)) == [
        '--model', 'rel/a.gguf']
    assert emit.cli(ctx({'in_imgs': ['a', 'b']}, rel=rel)) == [
        '--ref-image', 'rel/a', '--ref-image', 'rel/b']


# ---------------------------------------------------------------------------
# Kinds and sources
# ---------------------------------------------------------------------------

def test_flag_kind(monkeypatch):
    use(monkeypatch, Option('mmap', cli=Out('--mmap', kind='flag')))
    assert emit.cli(ctx({'in_mmap': True})) == ['--mmap']
    assert emit.cli(ctx({'in_mmap': False})) == []
    assert emit.cli(ctx({})) == []


def test_repeat_kind(monkeypatch):
    use(monkeypatch, Option('imat', cli=Out('--imat-in', kind='repeat')))
    assert emit.cli(ctx({'in_imat': ['a', 'b']})) == [
        '--imat-in', 'a', '--imat-in', 'b']
    assert emit.cli(ctx({'in_imat': []})) == []


def test_sources(monkeypatch):
    use(monkeypatch,
        Option('a', virtual=True, source='cfg:lora_dir', cli='--lora'),
        Option('b', virtual=True, source='ctx:out', cli='-o'),
        Option('c', source='f:model', cli='--model'),
        Option('d', cli=Out('--init-img', source=('img', 'frame'))))
    params = {'f_model': 'm.gguf', 'in_img': '', 'in_frame': 'f.png'}
    got = emit.cli(ctx(params, cfg={'lora_dir': 'loras'}.get,
                       values={'out': 'o.png'}))
    assert got == ['--lora', 'loras', '-o', 'o.png', '--model', 'm.gguf',
                   '--init-img', 'f.png']


def test_assembled_value_ignores_the_missing_setting(monkeypatch):
    use(monkeypatch, Option(
        'ref_image_args', virtual=True, omit=(),
        cli=Out('--ref-image-args', convert='ref_image_args')))
    assert emit.cli(ctx({})) == []
    assert emit.cli(ctx({'in_increase_ref_index': True})) == [
        '--ref-image-args', 'ref_index_mode=increase']


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('condition, params, expected', [
    ('on', {'in_on': True}, True),
    ('on', {'in_on': 0}, False),
    ('on', {}, False),
    ('!on', {}, True),
    ('!on', {'in_on': 1}, False),
    ('mode==1', {'in_mode': 1}, True),
    ('mode==1', {'in_mode': 0}, False),
    ('mode==1', {}, False),
    ('mode!=1', {}, True),
    ('mode!=1', {'in_mode': 1}, False),
    ('pad==Circular X', {'in_pad': 'Circular X'}, True),
    ('pad==Circular X', {'in_pad': 'Circular'}, False),
    ('?on', {'in_on': False}, True),
    ('?on', {}, False),
    ('f:model', {'f_model': 'a'}, True),
    ('f:model', {'in_model': 'a'}, False),
])
def test_conditions(condition, params, expected):
    assert emit.check(condition, ctx(params)) is expected


def test_runner_flag_condition():
    assert emit.check('@lora_tags', ctx({}, flags={'lora_tags': True}))
    assert not emit.check('@lora_tags', ctx({}))


def test_all_conditions_must_hold(monkeypatch):
    use(monkeypatch, Option('g', cli=Out('--g', when=('a',)), when=('b',)))
    assert emit.cli(ctx({'in_g': 1, 'in_a': 1, 'in_b': 1})) == ['--g', '1']
    assert emit.cli(ctx({'in_g': 1, 'in_a': 1})) == []
    assert emit.cli(ctx({'in_g': 1, 'in_b': 1})) == []


# ---------------------------------------------------------------------------
# Targets and scopes
# ---------------------------------------------------------------------------

def test_targets_are_independent(monkeypatch):
    use(monkeypatch, Option('steps', cli='--steps',
                            payload='sample_params.sample_steps'),
        Option('threads', cli='-t', launch='-t'))
    params = {'in_steps': 20, 'in_threads': 8}
    assert emit.cli(ctx(params)) == ['--steps', '20', '-t', '8']
    assert emit.launch(ctx(params, 'server')) == ['-t', '8']
    assert emit.payload(ctx(params)) == {
        'sample_params': {'sample_steps': 20}}


@pytest.mark.parametrize('scopes, kind, expected', [
    ((), 'txt2img', True), ((), 'video', True), ((), 'upscale', True),
    ((), 'convert', False),                      # cli default: 'all'
    (('image',), 'imgedit', True), (('image',), 'video', False),
    (('gen',), 'video', True), (('gen',), 'upscale', False),
    (('video',), 'video', True), (('video',), 'txt2img', False),
    (('img2img',), 'img2img', True), (('img2img',), 'txt2img', False),
    (('convert',), 'convert', True),
])
def test_cli_scopes(monkeypatch, scopes, kind, expected):
    use(monkeypatch, Option('x', cli=Out('--x', scopes=scopes)))
    assert bool(emit.cli(ctx({'in_x': 1}, kind))) is expected


def test_launch_and_payload_default_scopes(monkeypatch):
    use(monkeypatch, Option('x', launch='--x', payload='x'))
    assert emit.launch(ctx({'in_x': 1}, 'server')) == ['--x', '1']
    assert emit.launch(ctx({'in_x': 1}, 'txt2img')) == []
    assert emit.payload(ctx({'in_x': 1}, 'imgedit')) == {'x': 1}
    assert emit.payload(ctx({'in_x': 1}, 'video')) == {}


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def test_payload_nested_paths_and_base(monkeypatch):
    use(monkeypatch,
        Option('a', payload='sample_params.guidance.txt_cfg'),
        Option('b', payload=Out(path=('x.w', 'x.h'))))
    base = {'sample_params': {'sample_steps': 20}}
    got = emit.payload(ctx({'in_a': 7.0, 'in_b': 64}), base)
    assert got is base
    assert got == {'sample_params': {'sample_steps': 20,
                                     'guidance': {'txt_cfg': 7.0}},
                   'x': {'w': 64, 'h': 64}}


def test_payload_keeps_false_values(monkeypatch):
    use(monkeypatch, Option(
        'off', payload=Out(path='embed', convert='not', omit=(),
                           when=('?off',))))
    assert emit.payload(ctx({'in_off': True})) == {'embed': False}
    assert emit.payload(ctx({'in_off': False})) == {'embed': True}
    assert emit.payload(ctx({})) == {}


def test_payload_flag_kind(monkeypatch):
    use(monkeypatch, Option('t', payload=Out(path='tiling.temporal',
                                             kind='flag')))
    assert emit.payload(ctx({'in_t': True})) == {'tiling': {'temporal': True}}
    assert emit.payload(ctx({'in_t': False})) == {}


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

HIRES = Group('hires', toggle='hires_bool', requires_key=True,
              marker='hires.enabled')


def hires_options():
    return (
        Option('hires_scale', group='hires', cli='--hires-scale',
               payload=Out(path='hires.scale', convert='float')),
        Option('hires_always', group='hires',
               cli=Out('--always', gated=False),
               payload=Out(path='hires.always', gated=False)),
    )


def test_group_gates_the_cli(monkeypatch):
    use(monkeypatch, *hires_options(), groups=(HIRES,))
    params = {'in_hires_scale': 2.0, 'in_hires_always': 1}
    assert emit.cli(ctx({**params, 'in_hires_bool': True})) == [
        '--hires-scale', '2.0', '--always', '1']
    assert emit.cli(ctx({**params, 'in_hires_bool': False})) == [
        '--always', '1']


def test_group_payload_on_off_and_missing(monkeypatch):
    use(monkeypatch, *hires_options(), groups=(HIRES,))
    params = {'in_hires_scale': 2.0, 'in_hires_always': 1}
    assert emit.payload(ctx({**params, 'in_hires_bool': True})) == {
        'hires': {'enabled': True, 'scale': 2.0, 'always': 1}}
    assert emit.payload(ctx({**params, 'in_hires_bool': False})) == {
        'hires': {'enabled': False, 'always': 1}}
    # the tab has no HiRes settings: nothing is sent
    assert emit.payload(ctx(params)) == {}


def test_group_off_fields(monkeypatch):
    cache = Group('cache', toggle='cache_bool', requires_key=True,
                  off={'cache_mode': 'disabled'})
    use(monkeypatch, Option('cache_mode', group='cache',
                            payload='cache_mode'), groups=(cache,))
    assert emit.payload(ctx({'in_cache_bool': False,
                             'in_cache_mode': 'ucache'})) == {
        'cache_mode': 'disabled'}
    assert emit.payload(ctx({'in_cache_bool': True,
                             'in_cache_mode': 'ucache'})) == {
        'cache_mode': 'ucache'}


def test_group_without_requires_key_has_no_marker_and_stays_empty(
        monkeypatch):
    slg = Group('slg', toggle='slg_bool')
    use(monkeypatch, Option('slg_scale', group='slg',
                            payload='guidance.slg.scale'), groups=(slg,))
    assert emit.payload(ctx({'in_slg_scale': 2})) == {}
    assert emit.payload(ctx({'in_slg_bool': True, 'in_slg_scale': 2})) == {
        'guidance': {'slg': {'scale': 2}}}


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('value, expected', [
    (None, -1), ('', -1), (0, 0), (1234.0, 1234), ('7', 7),
])
def test_seed(value, expected):
    assert convert.convert('seed', value, None) == expected


@pytest.mark.parametrize('value, expected', [
    ('euler a', 'euler_a'), ('k_dpmpp_2m', 'dpmpp_2m'),
    ('DDIM', 'ddim_trailing'),
    ('res multistep', 'res_multistep'), (' ', None),
])
def test_sampler_api(value, expected):
    assert convert.convert('sampler_api', value, None) == expected


def test_number_lists():
    assert convert.parse_number_list("1.0, 0.5, 0") == [1.0, 0.5, 0.0]
    assert convert.parse_number_list("[7,8,9]", int) == [7, 8, 9]
    assert convert.parse_number_list("bad") is None
    assert convert.parse_number_list("") is None


def test_tables():
    c = ctx({})
    assert convert.backend_table(
        [['*', 'cuda0'], ['vae', 'cpu'], ['te', 'default']], c
    ) == 'cuda0,vae=cpu'
    assert convert.params_backend_table(
        [['*', 'cpu'], ['vae', 'disk']], c) == '*=cpu,vae=disk'
    assert convert.backend_table([['vae', 'disk']], c) is None
    assert convert.split_modes(
        [['diffusion', 'row'], ['vae', 'row']], c) == 'diffusion=row'
    assert convert.max_vram([['cuda0', '6'], ['*', '4']], c) == 'cuda0=6,4'
    assert convert.max_vram([['x', 'nan']], c) is None
    # a budget of 0 isn't passed
    assert convert.max_vram([['*', '0']], c) is None
    assert convert.max_vram([['*', '0'], ['cuda0', '6']], c) == 'cuda0=6'
    assert convert.max_vram([['cuda0', '0.0']], c) is None
    assert convert.backend_table(None, c) is None


def test_model_args():
    assert convert.model_args(None, ctx({})) is None
    c = ctx({'in_chroma_use_dit_mask': False, 'in_chroma_t5_mask_pad': 3.0,
             'in_pixart_vae_scale_factor': 0})
    assert convert.model_args(None, c) == (
        'chroma_use_dit_mask=false,chroma_t5_mask_pad=3')

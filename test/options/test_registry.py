"""The option registry (modules/options/registry.py): its invariants."""
import re
from pathlib import Path

import pytest

from modules.options import convert, emit, registry
from modules.options.registry import GROUPS, OPTIONS
from modules.options.spec import (
    ALL_KINDS, GEN_KINDS, IMAGE_KINDS, NO_DEFAULT, SCOPE_ALIASES,
    expand_scopes
)

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / 'modules/core'
KINDS = set(ALL_KINDS) | {'convert', 'server'}
TARGETS = ('cli', 'launch', 'payload')


def _core_source() -> str:
    return "\n".join(p.read_text() for p in CORE.rglob('*.py'))


def _outs():
    return [(o, t, out) for o in OPTIONS for t in TARGETS
            for out in getattr(o, t)]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_names_and_ui_keys_are_unique():
    assert len({o.name for o in OPTIONS}) == len(OPTIONS)
    assert len({o.ui_key for o in OPTIONS}) == len(OPTIONS)


def test_groups_exist():
    assert {o.group for o in OPTIONS if o.group} <= set(GROUPS)


@pytest.mark.parametrize('option, target, out', _outs(),
                         ids=lambda v: getattr(v, 'name', None)
                         or getattr(v, 'flag', None) or str(v))
def test_each_output_is_well_formed(option, target, out):
    if target == 'payload':
        assert out.path and not out.flag, "a payload output needs a path"
    else:
        assert out.flag and not out.path, f"a {target} output needs a flag"
    assert out.kind in ('value', 'flag', 'repeat')
    if target == 'payload':
        assert out.kind != 'repeat'
    assert not (out.relative and target == 'payload'), \
        "payload fields aren't paths relative to the executable"
    assert all(s in KINDS or s in SCOPE_ALIASES for s in out.scopes)
    if out.convert:
        assert out.convert.partition(':')[0] in convert.CONVERTERS


def test_scope_aliases():
    assert expand_scopes(('image',)) == set(IMAGE_KINDS)
    assert expand_scopes(('gen',)) == set(GEN_KINDS)
    assert expand_scopes(('all', 'convert')) == set(ALL_KINDS) | {'convert'}


def test_cli_and_payload_scopes_stay_in_their_target():
    """A request is for the image tabs or video, the launch for the server."""
    for option, target, out in _outs():
        kinds = expand_scopes(out.scopes) if out.scopes else None
        if target == 'payload' and kinds:
            assert kinds <= set(GEN_KINDS), option.name
        if target == 'launch' and kinds:
            assert kinds <= {'server', 'video'}, option.name
        if target == 'cli' and kinds:
            assert 'server' not in kinds, option.name


# ---------------------------------------------------------------------------
# Every setting is registered
# ---------------------------------------------------------------------------

def test_every_setting_the_core_reads_is_registered():
    """
    A new 'in_x'/'f_x' read in modules/core must get a registry entry
    (UI-only settings too), so the registry stays the full list of options.
    """
    keys = set(re.findall(r"""['"]((?:in|f)_\w+)['"]""", _core_source()))
    missing = keys - registry.ui_keys()
    assert not missing, f"register in registry.OPTIONS: {sorted(missing)}"


def test_conditions_and_sources_name_registered_settings():
    known = registry.ui_keys() | {
        f"in_{o.name}" for o in OPTIONS}
    names = set()
    for option, _, out in _outs():
        for cond in option.when + out.when:
            if cond.startswith('@'):
                continue
            cond = cond.lstrip('!?')
            names.add(re.split(r'[=!]=', cond)[0])
        sources = out.source if out.source is not None else option.source
        for src in (sources if isinstance(sources, tuple) else (sources,)):
            if src and not src.startswith(('ctx:', 'cfg:')):
                names.add(src)
    for g in GROUPS.values():
        names.add(g.toggle)
    for name in names:
        key = name.replace(':', '_', 1) if name.startswith('f:') \
            else f"in_{name}"
        assert key in known, f"'{name}' is not a registered setting"


def test_resolved_paths_come_from_a_model_folder():
    """'f:<name>' needs the option to say which folder it lives in."""
    by_name = {o.name: o for o in OPTIONS}
    used = set()
    for option, _, out in _outs():
        for cond in option.when + out.when:
            if 'f:' in cond:
                used.add(cond.lstrip('!?').split(':', 1)[1])
        for src in ((out.source or option.source),):
            for s in (src if isinstance(src, tuple) else (src,)):
                if s and s.startswith('f:'):
                    used.add(s[2:])
    for name in used:
        assert by_name[name].path_dir, f"'{name}' has no path_dir"


def test_path_dir_options_resolve_their_own_path():
    for o in OPTIONS:
        if o.path_dir:
            assert o.source == f'f:{o.name}', o.name
            assert o.path_dir.endswith('_dir')


def test_path_mappings_cover_the_model_folders():
    mappings = registry.path_mappings()
    assert 'in_ckpt_model' in mappings['ckpt_dir']
    assert 'in_taesd' in mappings['taesd_dir']
    assert set(mappings) <= {
        'ckpt_dir', 'vae_dir', 'unet_dir', 'txt_enc_dir', 'taesd_dir',
        'phtmkr_dir', 'upscl_dir', 'cnnet_dir', 'ad_dir', 'ipadapter_dir',
        'pulid_dir', 'animatediff_dir', 'tokenizer_dir'}


def test_a_condition_shared_by_all_outputs_is_on_the_option():
    """
    When every output of an option carries the same when=, it belongs to
    the option (Option.when), so it's written once.
    """
    repeated = []
    for o in OPTIONS:
        outs = [out for t in TARGETS for out in getattr(o, t)]
        if len(outs) > 1 and outs[0].when \
                and len({out.when for out in outs}) == 1:
            repeated.append(o.name)
    assert not repeated, f"move the shared when= to the option: {repeated}"


def test_what_the_registry_expects_from_the_runners_is_supplied():
    """
    'ctx:<name>' sources and '@<flag>' conditions are filled in by the
    runners: each one must be mentioned in modules/core.
    """
    source = _core_source()
    expected = set()
    for option, _, out in _outs():
        for src in (out.source if out.source is not None else option.source,):
            for s in (src if isinstance(src, tuple) else (src,)):
                if s and s.startswith('ctx:'):
                    expected.add(s[4:])
        expected.update(c[1:] for c in option.when + out.when
                        if c.startswith('@'))
    missing = {name for name in expected
               if f"'{name}'" not in source and f'{name}=' not in source}
    assert not missing, f"no runner supplies: {sorted(missing)}"


def test_the_core_has_no_hand_written_flags():
    """
    A registered flag must not also be written by hand in modules/core:
    the registry is the only place that knows what each option outputs.
    """
    source = _core_source()
    flags = {out.flag for _, t, out in _outs() if t != 'payload'}
    handwritten = {f for f in flags
                   if f"'{f}'" in source or f'"{f}"' in source}
    assert not handwritten, f"remove from modules/core: {sorted(handwritten)}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_holds_the_defaults_and_only_the_folders_are_left():
    from modules.settings import DEFAULT_SETTINGS, DIR_DEFAULTS
    assert all(k.endswith('_dir') for k in DIR_DEFAULTS)
    for o in OPTIONS:
        if o.virtual or o.default is NO_DEFAULT:
            assert o.config_key not in DEFAULT_SETTINGS, o.name
        else:
            assert DEFAULT_SETTINGS[o.config_key] == o.default, o.name


def test_key_map_is_generated():
    mapping = registry.config_key_map()
    assert mapping['in_rpc_servers'] == 'def_rpc_servers'
    assert 'in_steps' not in mapping            # registered as 'def_steps'
    key = 'env_CUDA_VISIBLE_DEVICES'
    assert mapping[key] == 'def_' + key
    assert all(not o.virtual for o in OPTIONS if o.ui_key in mapping)


def test_settings_tab_has_no_hand_written_map():
    text = (ROOT / 'modules/interfaces/common/settings_tab.py').read_text()
    assert not re.search(r"^\s*'in_\w+':\s*'def_", text, re.M)


def test_table_defaults_are_independent():
    tables = [o for o in OPTIONS if isinstance(o.default, list)]
    assert len({id(o.default) for o in tables}) == len(tables)


# ---------------------------------------------------------------------------
# What each runner kind gets
# ---------------------------------------------------------------------------

def _defaults_params():
    return {o.ui_key: o.default for o in OPTIONS
            if o.default is not NO_DEFAULT and not o.virtual}


@pytest.mark.parametrize('kind', sorted(KINDS))
def test_defaults_build_without_errors(kind):
    params = _defaults_params()
    context = emit.Context(params, kind)
    if kind == 'server':
        emit.launch(context)
    else:
        emit.cli(context)
    if kind in IMAGE_KINDS:
        emit.payload(context)


def test_defaults_leave_sdcpp_defaults_alone():
    """With the config defaults, only the flags that matter are output."""
    params = _defaults_params()
    for kind in ('txt2img', 'video'):
        tokens = emit.cli(emit.Context(params, kind))
        flags = _flags(tokens)
        # default modes/limits aren't passed
        for flag in ('--auto-fit', '--batched-cfg', '--rpc-servers',
                     '--log-level', '--compression-quality', '--type',
                     '--prediction', '--vae-format', '-t', '--flash-attn',
                     '--conv-direct', '--max-vram', '--split-mode',
                     '--rng', '--sampler-rng', '--clip-skip', '-b'):
            assert flag not in flags, (kind, flag)


def _flags(tokens):
    return {t for t in tokens if re.match(r'--?[A-Za-z]', t)}


def _pairs(tokens):
    """{flag: value} of the valued flags of a command."""
    return {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1)
            if re.match(r'--?[A-Za-z]', tokens[i])
            and not re.match(r'--?[A-Za-z]', tokens[i + 1])}


MODELS = {'--model', '--vae', '--diffusion-model', '--motion-module',
          '--high-noise-diffusion-model', '--t5xxl', '--llm'}


def _models(tokens):
    return {f: v for f, v in _pairs(tokens).items() if f in MODELS}


def test_server_launch_only_has_what_is_not_sent_per_request():
    params = {**_defaults_params(), 'in_ip': '127.0.0.1', 'in_port': 1234,
              'in_hires_bool': True, 'in_cache_bool': True,
              'in_vae_tiling': True, 'in_upscl_bool': True}
    dirs = {'emb_dir': 'e', 'lora_dir': 'l', 'upscl_dir': 'u'}
    launch = _flags(emit.launch(
        emit.Context(params, 'server', cfg=dirs.get)))
    for flag in ('--steps', '--cfg-scale', '-W', '-H', '-s', '--hires',
                 '--cache-mode', '--vae-tiling', '--compression-quality',
                 '--extra-sample-args', '--upscale-model', '--scheduler',
                 '--hires-scale', '--extra-tiling-args'):
        assert flag not in launch, flag
    assert {'--listen-ip', '--listen-port', '--embd-dir', '--lora-model-dir',
            '--hires-upscalers-dir', '--mmap', '--color'} <= launch


def test_listen_address_defaults():
    context = emit.Context({}, 'server')
    assert emit.launch(context) == ['--listen-ip', '127.0.0.1',
                                    '--listen-port', '1234']
    context = emit.Context({'in_ip': '0.0.0.0', 'in_port': 7861.0}, 'server')
    assert emit.launch(context) == ['--listen-ip', '0.0.0.0',
                                    '--listen-port', '7861']


def test_upscale_runs_only_the_runtime_options():
    params = {'in_threads': 8, 'in_steps': 20, 'in_predict': 'eps',
              'in_rpc_servers': 'h:1', 'in_mmap': True,
              'in_img_preprocess': 'canny', 'in_vae_tiling': True,
              'in_extra_tiling_args': 'a=1'}
    flags = _flags(emit.cli(emit.Context(params, 'upscale')))
    assert flags == {'--rpc-servers', '--mmap', '--vae-tiling',
                     '--extra-tiling-args'}


def test_video_only_options_stay_out_of_the_image_commands():
    params = {'in_frames': 33, 'in_fps': 16, 'in_temporal_tiling': True,
              'in_moe_boundary_bool': True, 'in_moe_boundary': 0.9}
    video_only = {'--video-frames', '--fps', '--temporal-tiling',
                  '--moe-boundary'}
    assert video_only <= _flags(emit.cli(emit.Context(params, 'video')))
    assert not video_only & _flags(emit.cli(emit.Context(params, 'txt2img')))


def test_animatediff_swaps_the_video_models():
    params = {'f_unet_model': 'wan.gguf', 'f_unet_vae': 'v.safetensors',
              'f_animatediff_ckpt': 'sd15.st', 'f_animatediff_vae': 'av.st',
              'f_motion_module': 'mm.st', 'in_video_mode': 1,
              'in_high_noise_bool': True, 'f_high_noise_model': 'hn.gguf'}
    animatediff = emit.Context(params, 'video')
    assert _models(emit.cli(animatediff)) == {
        '--model': 'sd15.st', '--vae': 'av.st', '--motion-module': 'mm.st'}
    animatediff.params['in_video_mode'] = 0
    assert _models(emit.cli(animatediff)) == {
        '--diffusion-model': 'wan.gguf', '--vae': 'v.safetensors',
        '--high-noise-diffusion-model': 'hn.gguf'}


def test_checkpoint_and_unet_models():
    params = {'in_diffusion_mode': 0, 'f_ckpt_model': 'a.safetensors',
              'f_unet_model': 'u.gguf', 'f_ckpt_vae': 'cv', 'f_unet_vae': 'uv'}
    ckpt = {'--model': 'a.safetensors', '--vae': 'cv'}
    unet = {'--diffusion-model': 'u.gguf', '--vae': 'uv'}
    for target, kind in ((emit.cli, 'txt2img'), (emit.launch, 'server')):
        assert _models(target(emit.Context(params, kind))) == ckpt
        params['in_diffusion_mode'] = 1
        assert _models(target(emit.Context(params, kind))) == unet
        params['in_diffusion_mode'] = 0


def test_the_scheduler_gives_way_to_custom_sigmas():
    params = {'in_scheduler': 'karras', 'in_sigmas': ''}
    assert '--scheduler' in emit.cli(emit.Context(params, 'txt2img'))
    params['in_sigmas'] = '1, 0.5'
    tokens = emit.cli(emit.Context(params, 'txt2img'))
    assert '--scheduler' not in tokens
    assert tokens[tokens.index('--sigmas') + 1] == '1, 0.5'


def test_payload_of_a_full_generation():
    params = {
        'in_sampling': 'Euler a', 'in_steps': 25.0, 'in_width': 768.0,
        'in_height': 512, 'in_cfg': 6.5, 'in_seed': None,
        'in_batch_count': 0, 'in_scheduler': 'normal', 'in_clip_skip': -1,
        'in_eta_bool': True, 'in_eta': 0.5,
        'in_hires_bool': True, 'in_hires_upscaler': 'models/up.pth',
        'in_hires_scale': 2.0,
        'in_cache_bool': False, 'in_cache_mode': 'ucache',
        'in_vae_tiling': True, 'in_vae_tile_size': 32,
        'in_vae_relative_bool': True, 'in_vae_relative_tile_size': 0.25,
        'in_disable_img_metadata': True, 'in_compression_quality': 90,
        'in_preview_bool': False,
    }
    payload = emit.payload(emit.Context(params, 'txt2img'),
                           {'sample_params': {}})
    assert payload == {
        'width': 768, 'height': 512, 'seed': -1, 'batch_count': 1,
        'clip_skip': -1, 'embed_image_metadata': False,
        'output_compression': 90, 'preview': 'none', 'preview_interval': 1,
        'sample_params': {
            'sample_method': 'euler_a', 'sample_steps': 25,
            'scheduler': 'discrete', 'eta': 0.5,
            'guidance': {'txt_cfg': 6.5},
        },
        'hires': {'enabled': True, 'upscaler': 'up', 'scale': 2.0},
        'cache_mode': 'disabled',
        'vae_tiling_params': {
            'enabled': True, 'tile_size_w': 32, 'tile_size_h': 32,
            'rel_size_w': 0.25, 'rel_size_h': 0.25,
        },
    }


def test_tabs_without_the_settings_send_nothing_for_them():
    payload = emit.payload(emit.Context({'in_steps': 20}, 'txt2img'),
                           {'sample_params': {}})
    assert 'hires' not in payload and 'vae_tiling_params' not in payload
    assert 'cache_mode' not in payload
    assert payload['preview'] == 'proj'     # the default projection preview
    assert payload['seed'] == -1


@pytest.mark.parametrize('name, flag, kind', [
    ('sampling', '--sampling-method', 'txt2img'),
    ('scheduler', '--scheduler', 'txt2img'),
    ('high_noise_sampling', '--high-noise-sampling-method', 'video'),
])
def test_default_sampler_and_scheduler_are_left_to_sdcpp(name, flag, kind):
    """Choosing 'Default' leaves the flag out; any other value is passed."""
    params = {f'in_{name}': 'Default', 'in_high_noise_bool': True}
    assert flag not in emit.cli(emit.Context(params, kind))
    params[f'in_{name}'] = 'karras'
    tokens = emit.cli(emit.Context(params, kind))
    assert tokens[tokens.index(flag) + 1] == 'karras'


def test_default_sampler_and_scheduler_are_not_in_the_payload():
    params = {'in_sampling': 'Default', 'in_scheduler': 'Default'}
    payload = emit.payload(emit.Context(params, 'txt2img'),
                           {'sample_params': {}})
    assert payload['sample_params'] == {}
    params = {'in_sampling': 'Euler a', 'in_scheduler': 'karras'}
    payload = emit.payload(emit.Context(params, 'txt2img'),
                           {'sample_params': {}})
    assert payload['sample_params'] == {'sample_method': 'euler_a',
                                        'scheduler': 'karras'}


def test_sampler_and_scheduler_default_to_default():
    from modules.settings import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS['def_sampling'] == 'Default'
    assert DEFAULT_SETTINGS['def_scheduler'] == 'Default'
    # The dropdowns list 'Default' first (their other choices come from
    # sd.cpp itself, so the source is checked, not the imported lists)
    constants = (ROOT / 'modules/ui/constants.py').read_text()
    for name, opt in (('SAMPLERS', 'samplers'), ('SCHEDULERS', 'schedulers')):
        assert re.search(
            rf"^{name} = \['Default'\] \+ "
            rf"sd_choices\.get_choices\(\"{opt}\"\)",
            constants, re.M), name


def test_vae_tile_size_default_is_in_pixels():
    """sd.cpp counts the tile size in image pixels: its default is 256."""
    from modules.settings import DEFAULT_SETTINGS
    assert DEFAULT_SETTINGS['def_vae_tile_size'] == 256
    tile = {'in_vae_tiling': True, 'in_vae_tile_size': 256}
    tokens = emit.cli(emit.Context(tile, 'txt2img'))
    assert tokens[tokens.index('--vae-tile-size') + 1] == '256x256'
    # 0 leaves it to sd.cpp, on the CLI and on the server
    zero = {'in_vae_tiling': True, 'in_vae_tile_size': 0}
    assert '--vae-tile-size' not in emit.cli(emit.Context(zero, 'txt2img'))
    payload = emit.payload(emit.Context(zero, 'txt2img'))
    assert 'tile_size_w' not in payload['vae_tiling_params']


def test_taesd_preview_only_cli_needs_previews_the_server_does_not():
    """
    --taesd-preview-only is a startup option of sd-cli and sd-server. The
    CLI only passes it while previews are on; the server tabs have no
    preview toggle, so the server always gets it.
    """
    for kind in ('txt2img', 'video'):
        on = {'in_preview_taesd': True, 'in_preview_bool': True}
        assert '--taesd-preview-only' in emit.cli(emit.Context(on, kind))
        off = {'in_preview_taesd': True, 'in_preview_bool': False}
        assert '--taesd-preview-only' not in emit.cli(emit.Context(off, kind))
        assert '--taesd-preview-only' not in emit.cli(
            emit.Context({'in_preview_taesd': True}, kind))
    for preview in ({}, {'in_preview_bool': False}, {'in_preview_bool': True}):
        server = emit.Context({'in_preview_taesd': True, **preview}, 'server')
        assert '--taesd-preview-only' in emit.launch(server)
    assert '--taesd-preview-only' not in emit.launch(
        emit.Context({'in_preview_taesd': False}, 'server'))
    # previews are chosen per request: it's a request field of nothing
    assert 'preview_taesd' not in str(emit.payload(
        emit.Context({'in_preview_taesd': True}, 'txt2img')))


def test_the_server_has_its_own_color_flag():
    """sd-server defines --color itself (log colors, on by default)."""
    assert '--color' in emit.launch(emit.Context({'in_color': True}, 'server'))
    assert '--color' not in emit.launch(
        emit.Context({'in_color': False}, 'server'))


@pytest.mark.parametrize('kind, target', [('txt2img', emit.cli),
                                          ('video', emit.cli),
                                          ('server', emit.launch)])
def test_max_vram_needs_its_checkbox_and_a_budget(kind, target):
    table = [['*', '8'], ['cuda0', '0']]
    off = emit.Context({'in_max_vram_bool': False,
                        'in_max_vram_table': table}, kind)
    assert '--max-vram' not in target(off)
    missing = emit.Context({'in_max_vram_table': table}, kind)
    assert '--max-vram' not in target(missing)
    on = emit.Context({'in_max_vram_bool': True,
                       'in_max_vram_table': table}, kind)
    tokens = target(on)
    # the budget of 0 is left out
    assert tokens[tokens.index('--max-vram') + 1] == '8'
    zero = emit.Context({'in_max_vram_bool': True,
                         'in_max_vram_table': [['*', '0']]}, kind)
    assert '--max-vram' not in target(zero)


def test_default_model_folders_come_from_the_registry():
    """The config checks default models against the registry's folders."""
    from modules.settings import ConfigManager, DEFAULT_SETTINGS, DIR_DEFAULTS
    assert ConfigManager.PATH_MAP == registry.default_model_folders()
    assert ConfigManager.PATH_MAP['def_ckpt'] == 'ckpt_dir'     # default_key
    assert ConfigManager.PATH_MAP['def_t5xxl'] == 'txt_enc_dir'  # def_<name>
    assert set(ConfigManager.PATH_MAP.values()) <= set(DIR_DEFAULTS)
    # a default model key is never also a real default
    assert not set(ConfigManager.PATH_MAP) & set(DEFAULT_SETTINGS)


def test_default_key_is_only_for_model_files():
    for o in OPTIONS:
        if o.default_key:
            assert o.path_dir, f"{o.name}: default_key is for model files"
            assert o.default_key.startswith('def_'), o.name
            assert o.config_key == o.default_key

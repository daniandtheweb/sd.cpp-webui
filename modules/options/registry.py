"""sd.cpp-webui - Option registry

Single source of truth for the sd.cpp options. Each Option (see spec.py)
says, in one place:

- its config default and Settings-tab mapping ('def_<name>', 'in_<name>')
- what it outputs on each target, with its conditions:
    cli      the sd-cli command (txt2img, img2img, imgedit, video,
             upscale, convert)
    launch   the sd-server startup command
    payload  the JSON sent to sd-server with every request

The runners don't write flags or payload fields: they run
modules/options/emit.py over this table. The UI widgets stay in
modules/ui/, keyed by 'in_<name>' and reading their default with
config.get('def_<name>').

The table (OPTIONS, below) is ordered by topic, and the order of the
entries is the order of the arguments in a command:

    MODELS          model files, quantization, --model-args
    GENERATION      prompts, sampling, guidance, skip layer guidance
    INPUTS          input images, reference images, ControlNet,
                    IP-Adapter, PhotoMaker, PuLID
    POST-PROCESSING upscaler, HiRes Fix, ADetailer
    FEATURES        VAE tiling, cache, preview
    VIDEO           frames, high noise model, MoE, VACE
    RUNTIME         devices, memory, compute, server address, environment
    OUTPUT          output files, metadata, console, model conversion
    INTERFACE       sd.cpp-webui settings that never reach sd.cpp

Adding an option:
1. Add an Option(...) entry to OPTIONS, in the part of its topic (UI-only
   settings too).
2. Say what it outputs: cli=, launch=, payload=. Conditions go in
   when=, the section toggle in group=, the runner kinds in scopes=.
3. Run the tests: they check that every setting the code reads is
   registered, and that each output has a place to go.

Each entry lists its keywords in the same order: default, config details
(mapped, virtual, path_dir, source), omit, group, when, then the outputs
(cli, launch, payload) and, if an option of the image tabs has nothing for
the server, no_server= with the reason.

Which options the server doesn't get: `python -m modules.options.report`.

What the runners still do (the registry can't know it): the output and
prompt file paths, the preview path, the input images and masks, the
LoRA extraction, and the HTTP exchange. They pass what they compute to
the emission as ctx.values / ctx.flags.

Config: an option with a default also gets 'def_<name>' in the config.
The Settings tab saves it under that key either through the generated
'in_<name>' -> 'def_<name>' map (mapped=True, the default when the option
has a default) or, for the widgets it registers by hand, directly under
'def_<name>' (mapped=False: steps, sampling, theme, ...).

Directory settings ('*_dir') are not options: they stay in
modules/settings/defaults.py. The words option / setting / default / config
are explained in docs/terminology.md.

This module must not import application modules: config.py imports it.
"""

from typing import Any, Dict, List

from modules.options.spec import NO_DEFAULT, Group, Option, Out

# Runner kinds, for 'scopes=' (see spec.py)
IMG = ('image',)        # txt2img, img2img, imgedit
GEN = ('gen',)          # IMG + video
ALL = ('all',)          # GEN + upscale
VID = ('video',)
UP = ('upscale',)
I2I = ('img2img',)
EDIT = ('imgedit',)
CONV = ('convert',)


def Flag(flag=None, **kw) -> Out:
    """A bare flag (sd-server payload: true), output when truthy."""
    return Out(flag, kind='flag', **kw)


def File(flag, **kw) -> Out:
    """A flag followed by a path relative to the executable."""
    return Out(flag, relative=True, **kw)


def P(path, **kw) -> Out:
    """A payload field."""
    return Out(path=path, **kw)


# Conditions shared by several options
PULID = ('pulid_bool', 'f:pulid_weights', 'f:pulid_embedding')
IP_ADAPTER = ('ip_adapter_bool', 'f:ip_adapter', 'ip_adapter_img')
ADETAILER = ('ad_bool', 'f:ad_model')
TILE_SIZE = ('vae_tiling_params.tile_size_w', 'vae_tiling_params.tile_size_h')
REL_TILE_SIZE = ('vae_tiling_params.rel_size_w',
                 'vae_tiling_params.rel_size_h')

# Sections with an on/off toggle: their options say group='<name>'
GROUPS: Dict[str, Group] = {g.name: g for g in (
    Group('slg', toggle='slg_bool'),
    Group('hires', toggle='hires_bool', requires_key=True,
          marker='hires.enabled'),
    Group('vae_tiling', toggle='vae_tiling', requires_key=True,
          marker='vae_tiling_params.enabled', scopes=GEN),
    Group('cache', toggle='cache_bool', requires_key=True,
          off={'cache_mode': 'disabled'}),
    Group('preview', toggle='preview_bool'),
    Group('high_noise', toggle='high_noise_bool'),
)}


def _backend_rows() -> List[List[str]]:
    """Default rows of the backend tables (a fresh list for each)."""
    return [[name, 'default'] for name in (
        '*', 'diffusion', 'te', 'clip_vision', 'vae', 'controlnet',
        'photomaker', 'upscaler', 'detector', 'audio_encoder')]


OPTIONS: tuple = (
    # =========================================================================
    # MODELS: Model files, quantization and model-specific arguments.
    # =========================================================================
    # -- Mode selectors (UI-only) ---------------------------------------------
    # diffusion_mode: 0 = checkpoint, 1 = UNET (image tabs)
    # video_mode: 0 = UNET, 1 = AnimateDiff (video tab)
    Option('diffusion_mode'),
    Option('video_mode'),
    Option('model_tab', default='checkpoint', mapped=False),

    # -- Checkpoint and UNET models -------------------------------------------
    Option('ckpt_model', path_dir='ckpt_dir', default_key='def_ckpt',
           source='f:ckpt_model', when=('diffusion_mode==0',),
           cli=File('--model', scopes=IMG), launch=File('--model')),
    Option('ckpt_vae', path_dir='vae_dir', source='f:ckpt_vae',
           when=('diffusion_mode==0',), cli=File('--vae', scopes=IMG),
           launch=File('--vae')),
    Option('unet_model', path_dir='unet_dir', default_key='def_unet',
           source='f:unet_model',
           cli=(File('--diffusion-model', scopes=IMG,
                     when=('diffusion_mode==1',)),
                File('--diffusion-model', scopes=VID,
                     when=('video_mode!=1',))),
           launch=(File('--diffusion-model', when=('diffusion_mode==1',)),
                   File('--diffusion-model', scopes=VID,
                        when=('video_mode!=1',)))),
    Option('unet_vae', path_dir='vae_dir', source='f:unet_vae',
           cli=(File('--vae', scopes=IMG, when=('diffusion_mode==1',)),
                File('--vae', scopes=VID, when=('video_mode!=1',))),
           launch=(File('--vae', when=('diffusion_mode==1',)),
                   File('--vae', scopes=VID, when=('video_mode!=1',)))),
    Option('uncond_unet_model', path_dir='unet_dir',
           default_key='def_uncond_unet', source='f:uncond_unet_model',
           when=('diffusion_mode==1',),
           cli=File('--uncond-diffusion-model', scopes=IMG),
           launch=File('--uncond-diffusion-model')),
    Option('clip_g', path_dir='txt_enc_dir', source='f:clip_g',
           when=('diffusion_mode==1',), cli=File('--clip_g', scopes=IMG),
           launch=File('--clip_g')),
    Option('clip_l', path_dir='txt_enc_dir', source='f:clip_l',
           when=('diffusion_mode==1',), cli=File('--clip_l', scopes=IMG),
           launch=File('--clip_l')),
    Option('t5xxl', path_dir='txt_enc_dir', source='f:t5xxl',
           when=('diffusion_mode==1',), cli=File('--t5xxl', scopes=IMG),
           launch=File('--t5xxl')),
    Option('llm', path_dir='txt_enc_dir', source='f:llm',
           cli=(File('--llm', scopes=IMG, when=('diffusion_mode==1',)),
                File('--llm', scopes=VID, when=('video_mode!=1',))),
           launch=(File('--llm', when=('diffusion_mode==1',)),
                   File('--llm', scopes=VID, when=('video_mode!=1',)))),
    Option('llm_vision', path_dir='txt_enc_dir', source='f:llm_vision',
           when=('diffusion_mode==1',), cli=File('--llm_vision', scopes=IMG),
           launch=File('--llm_vision')),
    Option('tokenizer', path_dir='tokenizer_dir', source='f:tokenizer',
           when=('diffusion_mode==1',), cli=File('--tokenizer', scopes=IMG),
           launch=File('--tokenizer')),

    # -- Video models ---------------------------------------------------------
    Option('umt5_xxl', path_dir='txt_enc_dir', source='f:umt5_xxl',
           when=('video_mode!=1',), cli=File('--t5xxl', scopes=VID),
           launch=File('--t5xxl', scopes=VID)),
    Option('clip_vision_h', path_dir='txt_enc_dir', source='f:clip_vision_h',
           when=('video_mode!=1',), cli=File('--clip_vision', scopes=VID),
           launch=File('--clip_vision', scopes=VID)),
    Option('audio_vae', path_dir='vae_dir', source='f:audio_vae',
           when=('video_mode!=1',), cli=File('--audio-vae', scopes=VID),
           launch=File('--audio-vae', scopes=VID)),
    Option('audio_encoder', path_dir='txt_enc_dir', source='f:audio_encoder',
           when=('video_mode!=1',), cli=File('--audio-encoder', scopes=VID),
           launch=File('--audio-encoder', scopes=VID)),
    Option('emb_connect', path_dir='txt_enc_dir', source='f:emb_connect',
           when=('video_mode!=1',),
           cli=File('--embeddings-connectors', scopes=VID),
           launch=File('--embeddings-connectors', scopes=VID)),

    # -- AnimateDiff ----------------------------------------------------------
    Option('animatediff_ckpt', path_dir='ckpt_dir',
           source='f:animatediff_ckpt', when=('video_mode==1',),
           cli=File('--model', scopes=VID),
           launch=File('--model', scopes=VID)),
    Option('animatediff_vae', path_dir='vae_dir', source='f:animatediff_vae',
           when=('video_mode==1',), cli=File('--vae', scopes=VID),
           launch=File('--vae', scopes=VID)),
    Option('motion_module', path_dir='animatediff_dir',
           source='f:motion_module', when=('video_mode==1',),
           cli=File('--motion-module', scopes=VID),
           launch=File('--motion-module', scopes=VID)),

    # -- Shared folders and models --------------------------------------------
    # No config default, but the Settings tab saves it through the key map.
    Option('taesd', mapped=True, path_dir='taesd_dir', source='f:taesd',
           cli=File('--taesd', scopes=GEN), launch=File('--taesd')),
    # Virtual: folders from the config.
    # TAESD only for the previews: the VAE still decodes the final image.
    # The CLI passes it only while previews are on (group: preview); the
    # server tabs have no preview toggle, so it's always passed there.
    Option('preview_taesd', default=False, group='preview',
           cli=Flag('--taesd-preview-only', scopes=GEN),
           launch=Flag('--taesd-preview-only', gated=False)),
    Option('embd_dir', virtual=True, source='cfg:emb_dir',
           cli=File('--embd-dir', scopes=GEN), launch=File('--embd-dir')),
    Option('lora_model_dir', virtual=True, source='cfg:lora_dir',
           cli=File('--lora-model-dir', scopes=GEN, when=('@lora_tags',)),
           launch=File('--lora-model-dir')),

    # -- Quantization and formats ---------------------------------------------
    Option('model_type', default='Default', omit=(None, '', 'Default'),
           cli=Out('--type', scopes=GEN), launch='--type'),
    Option('tensor_type_rules', default='',
           cli=Out('--tensor-type-rules', scopes=GEN + CONV),
           launch='--tensor-type-rules'),
    Option('predict', default='Default', omit=(None, '', 'Default'),
           cli=Out('--prediction', scopes=GEN), launch='--prediction'),
    Option('vae_format', default='Default', omit=(None, '', 'Default'),
           cli=Out('--vae-format', scopes=GEN), launch='--vae-format'),

    # -- Model arguments (--model-args) ---------------------------------------
    # Virtual: assembled from the settings below.
    Option('model_args', virtual=True, omit=(),
           cli=Out('--model-args', scopes=GEN, convert='model_args'),
           launch=Out('--model-args', convert='model_args')),
    # Settings assembled into --model-args.
    Option('chroma_use_dit_mask', default=True),
    Option('chroma_use_t5_mask', default=False),
    Option('chroma_t5_mask_pad', default=0),
    Option('qwen_image_zero_cond_t', default=False),
    Option('qwen_image_2_1_prefix_cache', default=False),
    Option('qwen_image_2_1_prefix_cache_type', default='auto'),
    Option('pixart_pos_embed_base_size', default=0),
    Option('pixart_interpolation_scale', default=0),
    Option('pixart_vae_scale_factor', default=0),

    # =========================================================================
    # GENERATION: Prompts, sampling and guidance.
    # =========================================================================
    # -- Prompts and LoRAs ----------------------------------------------------
    # The runners supply the prompt texts and the LoRAs.
    Option('pprompt',
           payload=P('prompt', source='ctx:prompt', omit=(), scopes=GEN)),
    Option('nprompt',
           payload=P('negative_prompt', source='ctx:negative_prompt', omit=(),
                     scopes=GEN)),
    Option('loras', virtual=True,
           payload=P('lora', source='ctx:loras', scopes=GEN)),
    Option('lora_strength', default=1.0, mapped=False),
    Option('lora_apply', default='auto',
           cli=Out('--lora-apply-mode', scopes=GEN, when=('@lora_tags',)),
           launch=Out('--lora-apply-mode', omit=(None, '', 'Default'))),

    # -- Sampling -------------------------------------------------------------
    Option('sampling', default='Default', mapped=False,
           omit=(None, '', 'Default'),
           cli=Out('--sampling-method', scopes=GEN),
           payload=P('sample_params.sample_method', convert='sampler_api',
                     scopes=GEN)),
    Option('scheduler', default='Default', mapped=False,
           omit=(None, '', 'Default'),
           cli=Out('--scheduler', scopes=IMG, when=('!sigmas',)),
           payload=P('sample_params.scheduler', convert='scheduler_api')),
    Option('sigmas', default='', cli=Out('--sigmas', scopes=IMG),
           payload=P('sample_params.custom_sigmas', convert='number_list')),
    Option('steps', default=20, mapped=False,
           cli=Out('--steps', scopes=GEN, convert='int'),
           payload=P('sample_params.sample_steps', convert='int', scopes=GEN)),
    Option('width', default=512, mapped=False,
           cli=Out('-W', scopes=GEN, convert='int'),
           payload=P('width', convert='int', fallback=512, scopes=GEN)),
    Option('height', default=512, mapped=False,
           cli=Out('-H', scopes=GEN, convert='int'),
           payload=P('height', convert='int', fallback=512, scopes=GEN)),
    Option('cfg', default=7.0, mapped=False,
           cli=Out('--cfg-scale', scopes=GEN),
           payload=P('sample_params.guidance.txt_cfg', convert='float',
                     scopes=GEN)),
    Option('seed', default=-1, mapped=False,
           cli=Out('-s', scopes=GEN, convert='seed', omit=()),
           payload=P('seed', convert='seed', omit=(), scopes=GEN)),
    Option('batch_count', default=1, mapped=False,
           cli=Out('-b', scopes=GEN, convert='int', omit=(None, '', 0, 1)),
           payload=P('batch_count', convert='int_or_1', omit=()),
           no_server='the video request has no batch count'),
    Option('clip_skip', default=-1, mapped=False,
           cli=Out('--clip-skip', scopes=GEN, convert='int',
                   omit=(None, '', -1)),
           payload=P('clip_skip', convert='int', scopes=GEN)),
    Option('rng', default='Default', omit=(None, '', 'Default'),
           cli=Out('--rng', scopes=GEN), launch='--rng'),
    Option('sampler_rng', default='Default', omit=(None, '', 'Default'),
           cli=Out('--sampler-rng', scopes=GEN), launch='--sampler-rng'),
    Option('extra_sample_args', default='', strip=True,
           cli='--extra-sample-args',
           payload=P('sample_params.extra_sample_args', scopes=GEN)),

    # -- Guidance and shifts --------------------------------------------------
    Option('guidance_bool', default=False, mapped=False),
    Option('guidance', default=3.5, mapped=False, when=('guidance_bool',),
           cli=Out('--guidance', scopes=IMG),
           payload=P('sample_params.guidance.distilled_guidance',
                     convert='float')),
    Option('img_cfg_bool'),
    Option('img_cfg', default=7.0, mapped=False, when=('img_cfg_bool',),
           cli=Out('--img-cfg-scale', scopes=I2I),
           payload=P('sample_params.guidance.img_cfg', convert='float')),
    Option('strength', default=0.75, mapped=False,
           cli=Out('--strength', scopes=I2I),
           payload=P('strength', scopes=I2I, convert='float')),
    Option('flow_shift_bool', default=False, mapped=False),
    Option('flow_shift', default=3.0, mapped=False, when=('flow_shift_bool',),
           cli=Out('--flow-shift', scopes=GEN),
           payload=P('sample_params.flow_shift', convert='float', scopes=GEN)),
    Option('timestep_shift_bool'),
    Option('timestep_shift', when=('timestep_shift_bool',),
           cli=Out('--timestep-shift', scopes=IMG),
           payload=P('sample_params.shifted_timestep', convert='int')),
    Option('eta_bool'),
    Option('eta', when=('eta_bool',), cli=Out('--eta', scopes=IMG),
           payload=P('sample_params.eta', convert='float')),

    # -- Skip layer guidance (group: slg) -------------------------------------
    Option('slg_bool'),
    Option('slg_scale', group='slg', cli=Out('--slg-scale', scopes=GEN),
           payload=P('sample_params.guidance.slg.scale', convert='float',
                     scopes=GEN)),
    Option('skip_layer_start', group='slg',
           cli=Out('--skip-layer-start', scopes=GEN),
           payload=P('sample_params.guidance.slg.layer_start', convert='float',
                     scopes=GEN)),
    Option('skip_layer_end', group='slg',
           cli=Out('--skip-layer-end', scopes=GEN),
           payload=P('sample_params.guidance.slg.layer_end', convert='float',
                     scopes=GEN)),
    Option('skip_layers', group='slg', cli=Out('--skip-layers', scopes=GEN),
           payload=P('sample_params.guidance.slg.layers', convert='int_list',
                     scopes=GEN)),

    # =========================================================================
    # INPUTS: Input images and the conditioning built from them.
    # =========================================================================
    # -- Input images and masks -----------------------------------------------
    Option('img_inp',
           cli=(File('--init-img', scopes=I2I),
                File('--init-img', scopes=VID + UP,
                     source=('img_inp', 'first_frame_inp'))),
           payload=P('init_image', scopes=I2I + VID, source='ctx:init_image')),
    Option('img_mask',
           payload=P('mask_image', scopes=I2I + EDIT,
                     source='ctx:mask_image')),
    Option('mask'),
    Option('mask_img'),
    Option('invert_mask'),
    # Virtual: the runner saves the mask to a file (img2img CLI).
    Option('mask_file', virtual=True, source='ctx:mask_path',
           cli=File('--mask', scopes=I2I),
           no_server='the server gets the mask in the payload (mask_image)'),
    Option('img_preprocess', default='', strip=True,
           cli=Out('--image-preprocess', scopes=GEN),
           payload=P('image_preprocess', scopes=GEN)),

    # -- Reference images -----------------------------------------------------
    Option('ref_img',
           cli=File('--ref-image', kind='repeat', scopes=EDIT,
                    convert='image_paths'),
           payload=P('ref_images', scopes=EDIT, source='ctx:ref_images')),
    Option('increase_ref_index'),
    Option('disable_auto_resize_ref_image'),
    # Virtual: assembled from the two settings above.
    Option('ref_image_args', virtual=True, omit=(),
           cli=Out('--ref-image-args', scopes=IMG, convert='ref_image_args'),
           payload=P('ref_image_args', convert='ref_image_args')),

    # -- Qwen Image Layered ---------------------------------------------------
    Option('qwen_layered_bool'),
    Option('qwen_image_layers'),
    # Virtual: the layer count when Qwen Layered is on.
    Option('qwen_layers_arg', virtual=True, omit=(),
           cli=Out('--qwen-image-layers', scopes=IMG, convert='qwen_layers'),
           payload=P('qwen_image_layers', convert='qwen_layers')),

    # -- ControlNet -----------------------------------------------------------
    Option('cnnet_bool'),
    Option('cnnet', path_dir='cnnet_dir', source='f:cnnet',
           cli=File('--control-net', scopes=GEN, when=('cnnet_bool',)),
           launch=File('--control-net')),
    Option('control_img', when=('cnnet_bool',),
           cli=File('--control-image', scopes=GEN),
           payload=P('control_image', source='ctx:control_image'),
           no_server='the video request has no ControlNet'),
    Option('control_strength',
           cli=Out('--control-strength', scopes=GEN, when=('cnnet_bool',)),
           payload=P('control_strength', convert='float',
                     when=('cnnet_bool', '@control_image')),
           no_server='the video request has no ControlNet'),

    # -- IP-Adapter -----------------------------------------------------------
    Option('ip_adapter_bool'),
    Option('ip_adapter', path_dir='ipadapter_dir', source='f:ip_adapter',
           cli=File('--ip-adapter', scopes=IMG, when=IP_ADAPTER),
           launch=File('--ip-adapter')),
    Option('ip_adapter_clip_vision', path_dir='txt_enc_dir',
           source='f:ip_adapter_clip_vision',
           cli=File('--clip_vision', scopes=IMG, when=IP_ADAPTER),
           launch=File('--clip_vision', when=('f:ip_adapter',))),
    Option('ip_adapter_img',
           cli=File('--ip-adapter-image', scopes=IMG, when=IP_ADAPTER),
           payload=P('ip_adapter_image', source='ctx:ip_adapter_image',
                     when=('ip_adapter_bool',))),
    Option('ip_adapter_strength',
           cli=Out('--ip-adapter-strength', scopes=IMG, when=IP_ADAPTER),
           payload=P('ip_adapter_strength', convert='float',
                     when=('ip_adapter_bool', '@ip_adapter_image'))),

    # -- PhotoMaker -----------------------------------------------------------
    Option('phtmkr_bool'),
    Option('phtmkr', path_dir='phtmkr_dir', source='f:phtmkr',
           when=('phtmkr_bool',), cli=File('--photo-maker', scopes=GEN),
           launch=File('--photo-maker')),
    Option('phtmkr_id', when=('phtmkr_bool',),
           cli=File('--pm-id-images-dir', scopes=GEN),
           launch=File('--pm-id-images-dir')),
    Option('phtmkr_emb', when=('phtmkr_bool',),
           cli=File('--pm-id-embed-path', scopes=GEN),
           launch=File('--pm-id-embed-path')),
    Option('phtmkr_strength', when=('phtmkr_bool',),
           cli=Out('--pm-style-strength', scopes=IMG, convert='int'),
           launch=Out('--pm-style-strength')),

    # -- PuLID ----------------------------------------------------------------
    Option('pulid_bool'),
    Option('pulid_weights', path_dir='pulid_dir', source='f:pulid_weights',
           when=PULID, cli=File('--pulid-weights', scopes=GEN),
           launch=File('--pulid-weights')),
    Option('pulid_embedding', path_dir='pulid_dir', source='f:pulid_embedding',
           when=PULID, cli=File('--pulid-id-embedding', scopes=GEN),
           launch=File('--pulid-id-embedding')),
    Option('pulid_id_weight', when=PULID,
           cli=Out('--pulid-id-weight', scopes=GEN),
           launch=Out('--pulid-id-weight')),

    # =========================================================================
    # POST-PROCESSING: What happens to the image once it is generated.
    # =========================================================================
    # -- Upscaler (also the standalone upscale tab) ---------------------------
    Option('upscl_bool'),
    Option('upscl', path_dir='upscl_dir', source='f:upscl',
           cli=(File('--upscale-model', scopes=GEN, when=('upscl_bool',)),
                File('--upscale-model', scopes=UP)),
           no_server='images are upscaled with a separate /upscale request'),
    Option('upscl_rep',
           cli=(Out('--upscale-repeats', scopes=IMG, convert='int',
                    when=('upscl_bool',)),
                Out('--upscale-repeats', scopes=VID, when=('upscl_bool',)),
                Out('--upscale-repeats', scopes=UP, convert='int')),
           no_server='images are upscaled with a separate /upscale request'),
    Option('upscl_tile_size',
           cli=(Out('--upscale-tile-size', scopes=IMG, convert='int',
                    when=('upscl_bool',)),
                Out('--upscale-tile-size', scopes=VID, when=('upscl_bool',)),
                Out('--upscale-tile-size', scopes=UP, convert='int')),
           no_server='images are upscaled with a separate /upscale request'),
    Option('init_width', cli=Out('-W', scopes=UP, convert='int')),
    Option('init_height', cli=Out('-H', scopes=UP, convert='int')),

    # -- HiRes Fix (group: hires) ---------------------------------------------
    Option('hires_bool', cli=Flag('--hires', scopes=IMG)),
    Option('hires_upscaler', default='Latent', group='hires',
           cli=Out('--hires-upscaler', scopes=IMG, convert='splitext0',
                   fallback='Latent'),
           payload=P('hires.upscaler', convert='stem', fallback='Latent')),
    # Virtual: a folder from the config.
    Option('hires_upscalers_dir', virtual=True, source='cfg:upscl_dir',
           group='hires', cli=File('--hires-upscalers-dir', scopes=IMG),
           launch=File('--hires-upscalers-dir', gated=False)),
    Option('hires_scale', default=2.0, group='hires',
           cli=Out('--hires-scale', scopes=IMG),
           payload=P('hires.scale', convert='float')),
    Option('hires_width', default=0, group='hires',
           cli=Out('--hires-width', scopes=IMG, convert='int'),
           payload=P('hires.target_width', convert='int')),
    Option('hires_height', default=0, group='hires',
           cli=Out('--hires-height', scopes=IMG, convert='int'),
           payload=P('hires.target_height', convert='int')),
    Option('hires_steps', default=0, group='hires',
           cli=Out('--hires-steps', scopes=IMG, convert='int'),
           payload=P('hires.steps', convert='int')),
    Option('hires_denoising_strength', default=0.7, group='hires',
           cli=Out('--hires-denoising-strength', scopes=IMG),
           payload=P('hires.denoising_strength', convert='float')),
    Option('hires_upscale_tile_size', default=128, group='hires',
           cli=Out('--hires-upscale-tile-size', scopes=IMG, convert='int'),
           payload=P('hires.upscale_tile_size', convert='int')),
    Option('hires_sigmas', default='', group='hires',
           cli=Out('--hires-sigmas', scopes=IMG),
           payload=P('hires.custom_sigmas', convert='number_list')),

    # -- ADetailer ------------------------------------------------------------
    Option('ad_bool'),
    Option('ad_model', path_dir='ad_dir', source='f:ad_model',
           cli=File('--ad-model', scopes=IMG, when=ADETAILER),
           launch=File('--ad-model')),
    Option('ad_prompt', default='', strip=True,
           cli=Out('--ad-prompt', scopes=IMG, when=ADETAILER),
           payload='ad_prompt'),
    Option('ad_nprompt', default='', strip=True,
           cli=Out('--ad-negative-prompt', scopes=IMG, when=ADETAILER),
           payload='ad_negative_prompt'),
    # Settings assembled into extra_ad_args (see below).
    Option('ad_confidence', default=0.3),
    Option('ad_denoising_strength', default=0.4),
    Option('ad_mask_blur', default=4),
    Option('ad_inpaint_padding', default=32),
    Option('ad_dilate_erode', default=4),
    Option('ad_mask_mode', default='none'),
    Option('ad_inpaint_width', default=512),
    Option('ad_inpaint_height', default=512),
    Option('ad_steps', default=0),
    Option('ad_cfg', default=-1.0),
    Option('ad_extra_args', default=''),
    # Virtual: assembled from the ad_* settings above.
    Option('extra_ad_args', virtual=True, omit=(),
           cli=Out('--extra-ad-args', scopes=IMG, when=ADETAILER,
                   convert='ad_args'),
           payload=P('extra_ad_args', convert='ad_args')),

    # =========================================================================
    # FEATURES: How the image is decoded, cached and previewed.
    # =========================================================================
    # -- VAE tiling (group: vae_tiling) ---------------------------------------
    Option('vae_tiling', default=False, cli=Flag('--vae-tiling')),
    Option('vae_tile_overlap', default=0.5, group='vae_tiling',
           cli=Out('--vae-tile-overlap', scopes=GEN),
           payload=P('vae_tiling_params.target_overlap', convert='float',
                     scopes=GEN)),
    # In image pixels (sd.cpp's own default is 256); 0 leaves it to sd.cpp
    Option('vae_tile_size', default=256, group='vae_tiling',
           cli=Out('--vae-tile-size', scopes=GEN, convert='tile_square',
                   omit=(None, '', 0)),
           payload=Out(path=TILE_SIZE, convert='int', omit=(None, '', 0),
                       scopes=GEN)),
    Option('vae_relative_bool', default=False),
    Option('vae_relative_tile_size', default=0, group='vae_tiling',
           when=('vae_relative_bool',),
           cli=Out('--vae-relative-tile-size', scopes=GEN,
                   convert='tile_square', omit=(None, '', 0)),
           payload=Out(path=REL_TILE_SIZE, convert='float', omit=(None, '', 0),
                       scopes=GEN)),
    Option('temporal_tiling', default=False, group='vae_tiling',
           cli=Flag('--temporal-tiling', scopes=VID, gated=False),
           payload=Flag(path='vae_tiling_params.temporal_tiling', gated=False,
                        scopes=GEN)),
    Option('extra_tiling_args', default='', strip=True, group='vae_tiling',
           cli=Out('--extra-tiling-args', gated=False),
           payload=P('vae_tiling_params.extra_tiling_args', gated=False,
                     scopes=GEN)),

    # -- Cache (group: cache) -------------------------------------------------
    Option('cache_bool', default=False),
    Option('cache_mode', default='easycache', group='cache',
           cli=Out('--cache-mode', scopes=IMG),
           payload=P('cache_mode', convert='text')),
    Option('cache_option', default='', group='cache',
           cli=Out('--cache-option', scopes=IMG, convert='strip_quotes'),
           payload=P('cache_option', convert='strip_quotes')),
    Option('scm_mask', default='', group='cache',
           cli=Out('--scm-mask', scopes=IMG),
           payload=P('scm_mask', convert='text')),
    Option('scm_policy', default='none', group='cache',
           cli=Out('--scm-policy', scopes=IMG, omit=(None, '', 'none')),
           payload=P('scm_policy_dynamic', convert='dynamic_policy')),

    # -- Preview (group: preview) ---------------------------------------------
    Option('preview_bool', default=False),
    Option('preview_mode', default='none', group='preview',
           cli=Out('--preview', scopes=IMG),
           payload=P('preview', convert='api_preview_mode', omit=(),
                     gated=False, scopes=GEN)),
    Option('preview_interval', default=1, group='preview',
           cli=Out('--preview-interval', scopes=IMG, convert='int'),
           payload=P('preview_interval', convert='api_preview_interval',
                     omit=(), gated=False, scopes=GEN)),
    # Virtual: the runner picks the preview file.
    Option('preview_path', virtual=True, source='ctx:preview_path',
           group='preview', cli=Out('--preview-path', scopes=IMG),
           no_server='the server serves the preview of the running job'),
    Option('preview_noisy', default=False, group='preview',
           cli=Flag('--preview-noisy', scopes=GEN),
           no_server='the server request only has preview mode and interval'),

    # =========================================================================
    # VIDEO: Settings of the video tab (Wan, LTX, VACE, ...).
    # =========================================================================
    # -- Frames and inputs ----------------------------------------------------
    Option('frames', cli=Out('--video-frames', scopes=VID, convert='int'),
           payload=P('video_frames', scopes=VID, convert='int')),
    Option('fps', cli=Out('--fps', scopes=VID),
           payload=P('fps', scopes=VID, convert='int')),
    Option('first_frame_inp'),
    Option('last_frame_inp', cli=File('--end-img', scopes=VID),
           payload=P('end_image', scopes=VID, source='ctx:end_image')),
    Option('control_video_dir', cli=File('--control-video', scopes=VID),
           payload=P('control_frames', scopes=VID,
                     source='ctx:control_frames')),
    Option('s2v_audio', cli=File('--audio', scopes=VID),
           payload=P('ref_audios', scopes=VID, source='ctx:ref_audios')),

    # -- High noise model (group: high_noise) ---------------------------------
    Option('high_noise_bool'),
    Option('high_noise_model', path_dir='unet_dir',
           source='f:high_noise_model', group='high_noise',
           when=('video_mode!=1',),
           cli=File('--high-noise-diffusion-model', scopes=VID),
           launch=File('--high-noise-diffusion-model', scopes=VID,
                       gated=False)),
    Option('high_noise_cfg', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-cfg-scale', scopes=VID),
           payload=P('high_noise_sample_params.guidance.txt_cfg', scopes=VID,
                     convert='float')),
    Option('high_noise_sampling', omit=(None, '', 'Default'),
           group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-sampling-method', scopes=VID),
           payload=P('high_noise_sample_params.sample_method', scopes=VID,
                     convert='sampler_api')),
    Option('high_noise_steps', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-steps', scopes=VID),
           payload=P('high_noise_sample_params.sample_steps', scopes=VID,
                     convert='int')),
    Option('high_noise_img_cfg', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-img-cfg-scale', scopes=VID),
           payload=P('high_noise_sample_params.guidance.img_cfg', scopes=VID,
                     convert='float')),
    Option('high_noise_guidance', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-guidance', scopes=VID),
           payload=P('high_noise_sample_params.guidance.distilled_guidance',
                     scopes=VID, convert='float')),
    Option('high_noise_slg_scale', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-slg-scale', scopes=VID),
           payload=P('high_noise_sample_params.guidance.slg.scale', scopes=VID,
                     convert='float')),
    Option('high_noise_skip_layer_start', group='high_noise',
           when=('video_mode!=1',),
           cli=Out('--high-noise-skip-layer-start', scopes=VID),
           payload=P('high_noise_sample_params.guidance.slg.layer_start',
                     scopes=VID, convert='float')),
    Option('high_noise_skip_layer_end', group='high_noise',
           when=('video_mode!=1',),
           cli=Out('--high-noise-skip-layer-end', scopes=VID),
           payload=P('high_noise_sample_params.guidance.slg.layer_end',
                     scopes=VID, convert='float')),
    Option('high_noise_eta', group='high_noise', when=('video_mode!=1',),
           cli=Out('--high-noise-eta', scopes=VID),
           payload=P('high_noise_sample_params.eta', scopes=VID,
                     convert='float')),
    Option('high_noise_skip_layers', group='high_noise',
           when=('video_mode!=1',),
           cli=Out('--high-noise-skip-layers', scopes=VID),
           payload=P('high_noise_sample_params.guidance.slg.layers',
                     scopes=VID, convert='int_list')),

    # -- MoE boundary and VACE ------------------------------------------------
    Option('moe_boundary_bool'),
    Option('moe_boundary', when=('moe_boundary_bool',),
           cli=Out('--moe-boundary', scopes=VID),
           payload=P('moe_boundary', scopes=VID, convert='float')),
    Option('vace_strength_bool'),
    Option('vace_strength', when=('vace_strength_bool',),
           cli=Out('--vace-strength', scopes=VID),
           payload=P('vace_strength', scopes=VID, convert='float')),

    # =========================================================================
    # RUNTIME: Devices, memory, compute, the server address, the environment.
    # =========================================================================
    # -- Devices and memory ---------------------------------------------------
    Option('max_vram_bool', default=False),
    Option('max_vram_table', default=[], omit=(), when=('max_vram_bool',),
           cli=Out('--max-vram', scopes=GEN, convert='max_vram'),
           launch=Out('--max-vram', convert='max_vram')),
    Option('backend_table', default=_backend_rows(), omit=(),
           cli=Out('--backend', scopes=GEN, convert='backend_table'),
           launch=Out('--backend', convert='backend_table')),
    Option('params_backend_table', default=_backend_rows(), omit=(),
           cli=Out('--params-backend', scopes=GEN,
                   convert='params_backend_table'),
           launch=Out('--params-backend', convert='params_backend_table')),
    Option('split_modes', default=[], omit=(),
           cli=Out('--split-mode', scopes=GEN, convert='split_modes'),
           launch=Out('--split-mode', convert='split_modes')),
    Option('offload_to_cpu', default=False, cli=Flag('--offload-to-cpu'),
           launch=Flag('--offload-to-cpu')),
    Option('auto_fit', default='on', omit=(None, '', 'on'), cli='--auto-fit',
           launch='--auto-fit'),
    Option('rpc_servers', default='', strip=True, cli='--rpc-servers',
           launch='--rpc-servers'),
    Option('mmap', default=True, cli=Flag('--mmap'), launch=Flag('--mmap')),
    Option('eager_load', default=False, cli=Flag('--eager-load'),
           launch=Flag('--eager-load')),
    Option('disable_prefetch', default=False, cli=Flag('--disable-prefetch'),
           launch=Flag('--disable-prefetch')),
    Option('disable_segmented_compute', default=False,
           cli=Flag('--disable-segmented-compute'),
           launch=Flag('--disable-segmented-compute')),
    Option('conditioning_cache_size', default=0, omit=(None, 0),
           cli='--conditioning-cache-size',
           launch='--conditioning-cache-size'),

    # -- Compute --------------------------------------------------------------
    Option('threads', default=0, omit=(None, 0, '0', ''),
           cli=Out('-t', scopes=GEN), launch='-t'),
    Option('flash_attention', default='auto', omit=(),
           cli=Out('--flash-attn', convert='mode_spec:flash_attention'),
           launch=Out('--flash-attn', convert='mode_spec:flash_attention')),
    Option('flash_attention_diffusion', default='auto'),
    Option('flash_attention_vae', default='auto'),
    Option('conv_direct', default='auto', omit=(),
           cli=Out('--conv-direct', convert='mode_spec:conv_direct'),
           launch=Out('--conv-direct', convert='mode_spec:conv_direct')),
    Option('conv_direct_diffusion', default='auto'),
    Option('conv_direct_vae', default='auto'),
    Option('batched_cfg', default='on', omit=(None, '', 'on'),
           cli='--batched-cfg', launch='--batched-cfg'),
    Option('sage_attn', default=False, cli=Flag('--sage-attn'),
           launch=Flag('--sage-attn')),
    Option('linear_scale', default=0, omit=(None, 0), cli='--linear-scale',
           launch='--linear-scale'),
    Option('attn_scale', default=0, omit=(None, 0), cli='--attn-scale',
           launch='--attn-scale'),
    Option('force_sdxl_vae_conv_scale', default=False,
           cli=Flag('--force-sdxl-vae-conv-scale'),
           launch=Flag('--force-sdxl-vae-conv-scale')),
    Option('upscale_conv_direct', default=False, mapped=False,
           cli=Flag('--diffusion-conv-direct'),
           launch=Flag('--diffusion-conv-direct')),
    Option('circular_padding',
           cli=(Flag('--circular', when=('circular_padding==Circular',)),
                Flag('--circularx', when=('circular_padding==Circular X',)),
                Flag('--circulary', when=('circular_padding==Circular Y',))),
           launch=(Flag('--circular', when=('circular_padding==Circular',)),
                   Flag('--circularx', when=('circular_padding==Circular X',)),
                   Flag('--circulary',
                        when=('circular_padding==Circular Y',)))),

    # -- Deprecated placement flags (use the backend tables instead) ----------
    Option('clip_on_cpu', default=False, cli=Flag('--clip-on-cpu'),
           launch=Flag('--clip-on-cpu')),
    Option('vae_on_cpu', default=False, cli=Flag('--vae-on-cpu'),
           launch=Flag('--vae-on-cpu')),
    Option('control_net_cpu', default=False, cli=Flag('--control-net-cpu'),
           launch=Flag('--control-net-cpu')),

    # -- sd-server address ----------------------------------------------------
    Option('ip', launch=Out('--listen-ip', convert='listen_ip', omit=())),
    Option('port',
           launch=Out('--listen-port', convert='listen_port', omit=())),

    # -- Environment variables ------------------------------------------------
    Option('env_vk_visible_override', default=False, prefix=''),
    Option('env_GGML_VK_VISIBLE_DEVICES', default=0, prefix=''),
    Option('env_cuda_visible_override', default=False, prefix=''),
    Option('env_CUDA_VISIBLE_DEVICES', default=0, prefix=''),
    Option('env_GGML_VK_DISABLE_COOPMAT', default=False, prefix=''),
    Option('env_GGML_VK_DISABLE_INTEGER_DOT_PRODUCT', default=False,
           prefix=''),

    # =========================================================================
    # OUTPUT: Output files, image metadata, console and the convert command.
    # =========================================================================
    # -- Output files ---------------------------------------------------------
    Option('output', default=''),
    Option('output_file', virtual=True, source='ctx:output_path',
           cli=Out('-o', scopes=ALL + CONV),
           no_server='the server returns the image, the runner saves it'),
    Option('output_format', virtual=True,
           payload=P('output_format', source='ctx:output_format', scopes=GEN)),
    Option('prompt_file', virtual=True, source='ctx:prompt_file',
           cli=Out('--prompt-file', scopes=GEN),
           no_server='the server gets the prompts in the payload'),
    Option('negative_prompt_file', virtual=True,
           source='ctx:negative_prompt_file',
           cli=Out('--negative-prompt-file', scopes=GEN),
           no_server='the server gets the prompts in the payload'),
    Option('output_begin_idx', default=0,
           cli=Out('--output-begin-idx', scopes=GEN, convert='int',
                   omit=(None, '', 0), when=('@indexed_output',)),
           no_server='applied by the runner when it names the saved images'),

    # -- Image metadata and encoding ------------------------------------------
    Option('disable_img_metadata', default=False,
           cli=Flag('--disable-image-metadata'),
           payload=P('embed_image_metadata', convert='not', omit=(),
                     when=('?disable_img_metadata',)),
           no_server='the video request has no metadata option'),
    Option('compression_quality', default=90, omit=(None, '', 90),
           cli='--compression-quality',
           payload=P('output_compression', convert='int', omit=(None, ''),
                     scopes=GEN)),

    # -- Console --------------------------------------------------------------
    # sd-server has its own --color (on by default): the flag only enables it
    Option('color', default=True,
           cli=(Flag('--color', scopes=ALL),
                Flag('--color', scopes=CONV, fallback=True)),
           launch=Flag('--color')),
    Option('log_level', default='info', strip=True, omit=(None, '', 'info'),
           cli=Out('--log-level', scopes=ALL + CONV), launch='--log-level'),

    # -- Model conversion (-M convert) ----------------------------------------
    Option('convert_model', virtual=True, source='ctx:model_path',
           cli=Out('--model', scopes=CONV)),
    Option('orig_model'),
    Option('model_dir'),
    Option('gguf_name'),
    Option('quant_type', cli=Out('--type', scopes=CONV)),
    Option('convert_name', cli=Flag('--convert-name', scopes=CONV)),
    Option('imatrix_in', cli=Out('--imat-in', kind='repeat', scopes=CONV)),

    # =========================================================================
    # INTERFACE: sd.cpp-webui settings that never reach sd.cpp.
    # =========================================================================
    # -- Look and feel --------------------------------------------------------
    Option('theme', default='default', mapped=False),
    Option('color_scheme', default='System', mapped=False),
    Option('gallery_sorting', default='Date (Oldest First)', mapped=False),

    # -- Generation -----------------------------------------------------------
    Option('high_entropy_seed', default=False, mapped=False),

    # -- Output naming --------------------------------------------------------
    Option('output_scheme', default='Sequential', mapped=False),
    Option('output_steps', default=False, mapped=False),
    Option('output_quant', default=False, mapped=False),
)


def _check_unique():
    names = [o.name for o in OPTIONS]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"Duplicate option registry entries: {dupes}")
    groups = {o.group for o in OPTIONS if o.group} - set(GROUPS)
    if groups:
        raise ValueError(f"Unknown option groups: {groups}")


_check_unique()


def config_defaults() -> Dict[str, Any]:
    """'def_<name>' -> default, merged into DEFAULT_SETTINGS."""
    return {o.config_key: o.default for o in OPTIONS
            if o.default is not NO_DEFAULT and not o.virtual}


def config_key_map() -> Dict[str, str]:
    """'in_<name>' -> 'def_<name>', used by the Settings tab."""
    return {o.ui_key: o.config_key for o in OPTIONS if o.mapped}


def path_mappings() -> Dict[str, List[str]]:
    """Config folder key -> the UI keys of the models inside it."""
    mappings: Dict[str, List[str]] = {}
    for o in OPTIONS:
        if o.path_dir:
            mappings.setdefault(o.path_dir, []).append(o.ui_key)
    return mappings


def default_model_folders() -> Dict[str, str]:
    """
    Config key of a default model -> the folder its file is in, used by the
    config to check the file still exists.
    """
    return {o.config_key: o.path_dir for o in OPTIONS if o.path_dir}


def ui_keys() -> set:
    """Every key a tab can pass: the UI settings and resolved paths."""
    keys = {o.ui_key for o in OPTIONS if not o.virtual}
    keys |= {o.resolved_key for o in OPTIONS if o.path_dir}
    return keys

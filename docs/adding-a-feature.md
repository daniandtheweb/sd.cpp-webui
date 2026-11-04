# Adding a new feature (an sd.cpp option)

This guide shows how to add a new stable-diffusion.cpp option, for example a
new `--stag-attention` flag, so it works in the CLI tabs, in the server tabs,
or in both.

(The words option, setting, default and config are explained in
[terminology](terminology.md).)

Adding an option is **two edits**:

1. one entry in the **option registry** (`modules/options/registry.py`) that
   says what the option outputs and when;
2. one **widget** in `modules/ui/` that lets the user set it.

The runners (`modules/core/`) don't need to change: they run the registry.

## How it works

```
  UI widget  ──►  params dict  ──►  registry (what to output, when)  ──►  sd.cpp
  in_<name>       {'in_x': 3}       modules/options/emit.py
                                            │
                                            ├─ cli      sd-cli command
                                            ├─ launch   sd-server startup command
                                            └─ payload  sd-server request (JSON)
```

| File | Role |
| --- | --- |
| `modules/options/registry.py` | The table of every option (**the file you edit**) |
| `modules/options/spec.py` | The data model: `Option`, `Out`, `Group` |
| `modules/options/emit.py` | The engine that turns params into arguments / JSON |
| `modules/options/convert.py` | Named value converters (`'int'`, `'seed'`, tables, ...) |
| `modules/options/report.py` | `python -m modules.options.report`: what each option reaches |
| `modules/ui/*.py` | The widgets, keyed `in_<name>` |
| `modules/core/` | The runners: they call the engine, they don't write flags |

The three **targets**:

| Target | What it is | Registry field |
| --- | --- | --- |
| `cli` | the `sd-cli` command (txt2img, img2img, imgedit, video, upscale, convert) | `cli=` |
| `launch` | the `sd-server` **startup** command (model files, devices, ...) | `launch=` |
| `payload` | the `sd-server` **request** JSON (per generation) | `payload=` |

## 0. Before you start: four questions

Look the option up in the sd.cpp source
([`examples/common/common.cpp`](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/common/common.cpp)
for the arguments,
[`examples/server/api.md`](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md)
for the request fields) or in `sd-cli -h` / `sd-server -h`.

| Question | Decides |
| --- | --- |
| Is it a bare flag or does it take a value? | `Flag(...)` or `Out(...)` |
| Which commands use it: images, video, upscale, convert? | `scopes=` |
| Does the server take it at **startup** or **per request**? | `launch=` or `payload=` |
| Which value means "let sd.cpp decide"? | `omit=` |

How to tell startup from request for the server:

- A **request field** exists only if the server's request JSON has a key for
  it (for example `sample_params.sample_steps`). Use `payload=`.
- Otherwise the option may still be a **startup argument**: `sd-server`
  accepts the context options (models, devices, `--taesd-preview-only`, ...)
  and the server's own options (`--listen-ip`, `--color`, ...). Use `launch=`.
- Some options are **CLI-only** (`-o`, `--output-begin-idx`, `--preview-noisy`,
  ...). Give them `cli=` only and say why with `no_server=` (see
  [step 4](#4-check-what-you-got)).

## 1. Add the registry entry

Open `modules/options/registry.py`, find the part of your topic (MODELS,
GENERATION, INPUTS, POST-PROCESSING, FEATURES, VIDEO, RUNTIME, OUTPUT,
INTERFACE) and add an `Option(...)`. **The order of the entries is the order of
the arguments in a command**, so put it where you want it to appear.

### A bare flag, in the CLI and at server startup

```python
Option('stag_attention', default=False,
       cli=Flag('--stag-attention', scopes=GEN),
       launch=Flag('--stag-attention')),
```

`default` is the config default (`def_stag_attention`). `Flag` outputs the bare
flag when the setting is truthy. `scopes=GEN` means the image and video
commands (without a scope, `cli` applies to the image, video and upscale
commands, not to convert; `launch` always applies to the server).

### A value with a checkbox and a request field

```python
Option('stag_bool', default=False),                  # the checkbox
Option('stag_scale', default=1.0, when=('stag_bool',),
       cli=Out('--stag-scale', scopes=GEN),
       payload=P('sample_params.stag_scale', convert='float')),
```

`when=` lists conditions (all must hold). Here the scale is only output while
the checkbox is on, in every output of the option. The checkbox has no output of
its own: it is a plain option so it gets a config default and is registered.

### Several options behind one checkbox: a `Group`

When several options share a toggle, or the request needs an `enabled` marker or
an "off" state, use a group (HiRes, Cache, VAE tiling and SLG are groups):

```python
# in GROUPS, near the top of registry.py
Group('stag', toggle='stag_bool', requires_key=True,
      marker='stag.enabled'),

# in OPTIONS
Option('stag_bool', default=False),
Option('stag_scale', default=1.0, group='stag',
       cli=Out('--stag-scale', scopes=GEN),
       payload=P('stag.scale', convert='float')),
Option('stag_layers', default='', group='stag',
       cli=Out('--stag-layers', scopes=GEN),
       payload=P('stag.layers', convert='int_list')),
```

Members are only output while the toggle is on. In the request the group also
sets `stag.enabled` to true/false, and sends nothing when the tab has no such
checkbox (`requires_key=True`). For one option behind one checkbox, a plain
`when=` is enough (that is what Max VRAM does).

### Keep it tidy

- Keywords go in this order: `default`, config details (`mapped`, `virtual`,
  `path_dir`, `source`), `omit`, `group`, `when`, then `cli`, `launch`,
  `payload`, and `no_server` last.
- A condition shared by all the outputs of an option goes in the option's
  `when=`, once (a test checks it). Put it on a single output only when that
  output differs.
- Keep lines to 79 columns, like the rest of the registry.

## 2. Add the widget

The widget's key must be `in_<name>`, and it reads its default from
`config.get('def_<name>')`. For example in `modules/ui/performance.py`:

```python
stag_attention = gr.Checkbox(
    label="STAG attention",
    value=config.get('def_stag_attention')
)
...
return {
    ...
    'in_stag_attention': stag_attention,
}
```

To make a checkbox switch other widgets on and off, copy the pattern in
`modules/ui/cache.py`: set `interactive=` from the config and wire
`checkbox.change(partial(update_interactivity, n), ...)`.

### Which tabs get the widget

- **A widget in an existing block** (performance, extras, preview, ...) needs
  nothing else: the CLI tabs, the server tabs and the Settings tab already
  collect that block's dictionary, and the config default and saving come from
  the registry's `default=`.
- **A new block** has to be hooked up where it is used:
  - each tab that should show it: `inputs_map.update(new_block)` (see
    `modules/interfaces/cli/txt2img_tab.py`);
  - the Settings tab: add it to the list in the "Advanced settings" tab of
    `modules/interfaces/common/settings_tab.py`, so it can be saved as a default;
  - **server tabs, for startup (`launch=`) options**: add its keys to
    `_server_config_keys` in `modules/interfaces/server/*_tab.py`. These are
    the widgets that are locked while the server is running.

### Defaults and saving

`default=` in the registry is the only place a default is written. Existing
`config.json` files keep what they saved; a new key is added with its default
the next time the config loads. Settings the Settings tab registers one by one
under `def_<name>` (like `steps` or `theme`) say `mapped=False` in the registry;
settings that come from a reused block don't.

## 3. When the registry can't know a value

Some values are computed by the runner: the output path, prompt files, encoded
images, LoRAs. The registry names them, and the runner supplies them.

- In the entry, take the value from the runner: `source='ctx:output_path'`.
- In the runner, return it from `_run_values()` (CLI,
  `modules/core/cli/sdcpp_cli.py`) or from the `values` dict of
  `ApiTaskRunner._build_payload()` (server,
  `modules/core/server/sdcpp_server.py`).
- Conditions the runner computes work the same way: `when=('@lora_tags',)`,
  set through `flags=` in `_context(...)`.

Examples to copy: `mask_file` (a CLI file the runner saves) and
`control_img` (an encoded image in the request). A test fails if the registry
asks for a `ctx:` value or an `@` flag that no runner supplies.

## 4. Check what you got

```bash
python -m modules.options.report stag_attention
```

```
stag_attention  (both)
  ui key      in_stag_attention
  sd-cli      --stag-attention   [img2img, imgedit, txt2img, video]
  sd-server   --stag-attention   [startup command]
```

`python -m modules.options.report` with no argument lists every option by status:

| Status | Meaning |
| --- | --- |
| `both` | output on the CLI and on the server |
| `cli only` | nothing for the server, with the reason |
| `server only` | only the server gets it |
| `toggle` | a checkbox that only switches other options |
| `ui only` | not passed to sd.cpp (a config setting, ...) |

An option of the image tabs that has CLI output but nothing for the server
**must say why**, or a test fails:

```python
Option('preview_noisy', default=False, group='preview',
       cli=Flag('--preview-noisy', scopes=GEN),
       no_server='the server request only has preview mode and interval'),
```

Options for the video, upscale and convert commands are CLI-only by nature and
need no reason.

## 5. Test it

```bash
python -m pytest test -q                 # everything
python -m pytest test/options -q         # only the option tests
```

If you use `ruff`, `ruff check --select F modules/options test/options` finds
unused imports.

The app and the tests look for `sd-cli` and `sd-server` (in the working
directory or on `PATH`) and run `<exe> --version`. For tests a placeholder
script that exits 0 is enough.

Without writing anything, the existing tests already check that:

- every `in_*` / `f_*` key the code in `modules/core/` reads is registered;
- no registered flag is also written by hand in `modules/core/`;
- every output **fires** on each runner it targets (CLI kinds, server startup,
  server request), so a typo in a scope, a condition or a converter is caught;
- every image option reaches the server or says why (`no_server=`);
- config defaults, the Settings-tab key map and the groups are consistent.

Add a test for the behaviour that is yours. They live in
`test/options/test_registry.py` and use the engine directly:

```python
from modules.options import emit

def test_stag_attention():
    on = emit.Context({'in_stag_attention': True}, 'txt2img')
    assert '--stag-attention' in emit.cli(on)
    server = emit.Context({'in_stag_attention': True}, 'server')
    assert '--stag-attention' in emit.launch(server)
    assert '--stag-attention' not in emit.launch(
        emit.Context({'in_stag_attention': False}, 'server'))
```

`emit.Context(params, kind)` takes the parameters as the UI gives them and the
runner kind (`txt2img`, `img2img`, `imgedit`, `video`, `upscale`, `convert` for
the CLI; `server` for the startup command; the image kinds for the request).
Use `emit.payload(context, {'sample_params': {}})` for the request.

## Reference

### `Option`

| Field | Meaning |
| --- | --- |
| `name` | base name; the UI key is `in_<name>`, the config key `def_<name>` |
| `default` | the config default (leave it out for a UI-only setting) |
| `cli`, `launch`, `payload` | what it outputs (an `Out`, a list of them, or a string: a flag for `cli`/`launch`, a path for `payload`) |
| `omit` | values that mean "let sd.cpp decide" (not output). Default `(None, '')`. Booleans and numbers are kept apart (`False != 0`) |
| `when` | conditions for every output |
| `group` | name of its `Group` |
| `strip` | strip whitespace from a string value first |
| `virtual` | not a UI setting: it only outputs something (a config value, or one assembled from other settings). An option assembled by a converter says `omit=()` |
| `source` | where the value comes from instead of `in_<name>` (see below) |
| `prefix` | UI key prefix; `''` for the environment variables |
| `mapped` | how the Settings tab finds its config key (see `spec.py`) |
| `path_dir` | config folder key for model files; makes `f_<name>` resolve from it |
| `no_server` | why an image option has no server output |

### `Out` (one output)

| Field | Meaning |
| --- | --- |
| `flag` / `path` | the CLI flag, or the dotted request path (`a.b.c`; a tuple sets several fields) |
| `kind` | `'value'` (default), `'flag'` (bare, when truthy), `'repeat'` (the flag once per list item) |
| `scopes` | the runners it applies to (see below) |
| `when` | extra conditions |
| `omit` | overrides the option's `omit` (`()` omits nothing) |
| `convert` | a named converter (see below) |
| `fallback` | the value when the setting is missing or empty |
| `source` | overrides the option's `source` |
| `relative` | make the path relative to the executable (`File(...)` does this) |
| `gated` | `False`: output even when the option's group is off |

Shorthands: `Flag('--x')` is a bare flag, `File('--x')` a flag followed by a
relative path, `P('a.b')` a request field.

### Scopes

| Scope | Runners |
| --- | --- |
| `IMG` (`'image'`) | txt2img, img2img, imgedit |
| `GEN` (`'gen'`) | `IMG` + video |
| `ALL` (`'all'`) | `GEN` + upscale |
| `VID`, `UP`, `I2I`, `EDIT`, `CONV` | video, upscale, img2img only, imgedit only, the convert command |

Defaults: `cli` → the image, video and upscale commands (not convert),
`launch` → the server, `payload` → the image tabs.

### Conditions (`when=`)

| Condition | Holds when |
| --- | --- |
| `'name'` | the setting `in_<name>` is truthy (`'f:name'`: the resolved model path `f_<name>`) |
| `'!name'` | it is falsy or missing |
| `'name==1'`, `'name!=1'` | it equals / doesn't equal a number or string |
| `'?name'` | the tab has the setting at all |
| `'@flag'` | the runner set `flag` (see step 3) |

### Sources (`source=`)

`'name'` (another setting), `'f:name'` (a resolved model path), `'cfg:key'`
(a config value), `'ctx:name'` (computed by the runner), or a tuple: the first
truthy one wins.

### Converters (`convert=`)

| Name | Does |
| --- | --- |
| `int`, `float`, `text` | number from a Gradio value; stripped string (`None` if empty) |
| `not` | inverts a boolean |
| `seed` | `None` / `''` → `-1` (sd.cpp's random seed) |
| `int_or_1` | an integer, with `None` or `0` → `1` (the batch count) |
| `sampler_api`, `scheduler_api` | UI names → the server's names |
| `number_list`, `int_list` | `"1, 2"` / `"[7,8]"` → list of numbers / integers |
| `tile_square` | `256` → `'256x256'` (0 → left out) |
| `stem`, `splitext0` | file name without extension (with / without directories) |
| `strip_quotes`, `dynamic_policy` | cache option and SCM policy handling |
| `image_paths` | Gradio gallery value → list of paths |
| `backend_table`, `params_backend_table`, `split_modes`, `max_vram` | the device tables |
| `mode_spec:<name>` | `on\|off\|auto` plus per-module overrides (flash attention, conv direct) |
| `model_args`, `ad_args`, `ref_image_args`, `qwen_layers` | one value assembled from several settings |
| `api_preview_mode`, `api_preview_interval` | the request's preview fields |
| `listen_ip`, `listen_port` | the server address with its defaults |

A new converter is a function `(value, ctx)` in `modules/options/convert.py`,
registered in `CONVERTERS`. It returns the value to output, or `None` to leave it
out. It must not import application modules; `ctx.params` gives it the whole
parameter dict.

## Common mistakes

| Symptom | Cause and fix |
| --- | --- |
| Test: `register in registry.OPTIONS: [...]` | code in `modules/core/` reads an `in_*` key that has no entry |
| Test: `remove from modules/core: [...]` | a registered flag is also written by hand in the runners; delete the hand-written one |
| Test: `no server output and no no_server= reason: [...]` | give it `launch=` / `payload=`, or `no_server='why'` |
| Test: `move the shared when= to the option` | every output repeats the same condition: write it once in the option's `when=` |
| Test: `no runner supplies: [...]` | a `ctx:` source or `@` condition that no runner sets (step 3) |
| Test `...fires_on_its_runner_kinds` fails | the output can never fire: a condition that can't hold, a wrong scope, or a converter that returns `None`. An option assembled by a converter with no value of its own needs `omit=()` |
| A default `0` / `False` is passed to sd.cpp | add it to `omit=`, e.g. `omit=(None, '', 0)` |
| A value shows as `'None'` | it should be omitted: it is missing from the tab, so check the scope and the `when=` |
| Saved settings don't apply | `mapped=` doesn't match how `settings_tab.py` registers the widget |

When in doubt, copy a similar existing entry: `sage_attn` (a bare flag),
`guidance` (checkbox + value + request field), `slg_*` and `hires_*` (groups),
`rng` (a "Default" choice that is omitted), `max_vram_table` (a value built
from a table), `ckpt_model` (a model file with a condition).

## Checklist

1. `Option(...)` in `registry.py`, in the part of its topic, with `cli` /
   `launch` / `payload`, `scopes`, `when`, `omit`, `convert`.
2. A widget keyed `in_<name>` reading `config.get('def_<name>')`.
3. The widget reaches its tabs (existing block: nothing; new block: tabs,
   Settings tab, `_server_config_keys`).
4. `python -m modules.options.report <name>` shows what you meant.
5. A test for the behaviour, and `python -m pytest test -q` passes.

# Terminology: option, setting, default, config, choices

The code uses a few words with one meaning each. Use them the same way in new
code, comments and the UI.

| Word | Means | Where it lives |
| --- | --- | --- |
| **option** | one stable-diffusion.cpp parameter: a command-line flag and/or a request field | the option registry, `modules/options/registry.py` |
| **setting** | a value the user can change in the UI: an option's value, a webui preference (theme, output naming) or a folder | the widgets in `modules/ui/` |
| **default** | a setting's saved starting value | the config, under `def_<name>` |
| **config** | the saved file and the code that reads and writes it | `user_data/config.json`, `ConfigManager` in `modules/settings/store.py` |
| **choices** | what sd.cpp says is valid for some arguments (samplers, schedulers, preview modes, ...) | `sd_choices` (`SDChoicesCache` in `modules/utils/sd_interface.py`) |
| **params** | the dict of current setting values a tab hands to a runner | `self.params` in the runners |

## How they relate

```
 option            registry entry: what sd.cpp accepts, and what to output
   │
   ▼
 setting           the widget the user changes          params['in_steps']
   │  "Save defaults" (Settings tab)
   ▼
 default           the saved starting value             config 'def_steps'
   │
   ▼
 config            user_data/config.json                config.get('def_steps')
```

The registry also lists settings that never reach sd.cpp (toggles, mode
selectors, the theme, output naming) so that they get a default and a widget
key like the others. `python -m modules.options.report` shows them as
`ui only` or `toggle`.

## Keys

| Key | Means |
| --- | --- |
| `in_<name>` | the current value of a setting: the widget's key, and the key in `params` |
| `def_<name>` | its saved default, in the config |
| `f_<name>` | a model setting resolved to a full path by the runner |
| `<x>_dir` | a folder (models, outputs), in the config |

The model defaults are saved under the name of the setting without a prefix
(`def_t5xxl`), or under a short name where it was historically different
(`def_ckpt` for `ckpt_model`; the registry's `default_key=` records it). The
config uses these keys to check that the model file still exists.

## Where things live

| Place | Role |
| --- | --- |
| `modules/options/` | the options: the registry, the engine that turns them into arguments, the converters, the report |
| `modules/settings/` | what is saved: `defaults.py` (what a fresh config starts with) and `store.py` (`ConfigManager`) |
| `modules/shared_instance.py` | the shared `config` and `sd_choices` instances |
| `modules/interfaces/common/settings_tab.py` | the **Settings** tab, which edits the defaults (**Save defaults**, **Restore defaults**) |
| `modules/ui/` | the widgets, one block per feature |

## Adding code

- A new sd.cpp parameter is a new **option**: see
  [Adding a new feature](adding-a-feature.md).
- A new preference of the webui itself (not passed to sd.cpp) is a setting
  too: register it the same way, with no `cli=` / `launch=` / `payload=`, so it
  gets a default and a widget key.
- A new folder goes in `DIR_DEFAULTS` in `modules/settings/defaults.py`.
- Read saved values through `config.get(...)`; don't read `config.json`
  directly.

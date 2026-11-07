"""The params a generation hook gets."""

from modules.extensions.host import run_post, run_pre
from modules.extensions.loader import load_extensions


def test_pre_hooks_change_the_params_and_post_hooks_only_read(
    make_ext, capsys
):
    cfg = make_ext({'g.py': (
        "def register(api):\n"
        "    api.add_hook('pre_generation', pre)\n"
        "    api.add_hook('post_generation', post)\n"
        "def pre(ctx): ctx.params['seen'] = 1\n"
        "def post(ctx): ctx.params['late'] = 1\n"
    )})
    load_extensions(cfg)
    params = {}
    run_pre(params, 'txt2img')
    assert params == {'seen': 1}
    run_post(None, params, 'txt2img')
    assert params == {'seen': 1}
    assert "does not support item assignment" in capsys.readouterr().out

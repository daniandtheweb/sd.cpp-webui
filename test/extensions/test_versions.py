"""The API version: mandatory, and one the app has."""

import sys

import pytest

from modules.extensions.host import extensions
from modules.extensions.loader import load_extensions


def test_api_version_is_mandatory(make_ext):
    cfg = make_ext({'none.py': "def register(api): pass\n"},
                   declare_api=False)
    assert load_extensions(cfg) == []
    (ext,) = extensions()
    assert ext.status == 'failed'
    assert 'no API_VERSION' in ext.reason and 'API_VERSION = 1' in ext.reason


@pytest.mark.parametrize('declared, status, reason', [
    ('API_VERSION = 99', 'skipped', 'targets API v99, this app supports up to v1'),
    ('API_VERSION = -1', 'failed', 'API v-1 does not exist'),
    ('API_VERSION = 1.0', 'failed', 'no API_VERSION'),
])
def test_an_extension_can_only_declare_a_version_that_exists(
    make_ext, declared, status, reason
):
    cfg = make_ext({'x.py': declared + "\ndef register(api): pass\n"})
    load_extensions(cfg)
    (ext,) = extensions()
    assert (ext.status, reason in ext.reason) == (status, True), ext.reason


def test_the_api_reports_its_declared_version(make_ext):
    cfg = make_ext({'v.py': ("API_VERSION = 1\n"
                             "SEEN = []\n"
                             "def register(api):\n"
                             "    SEEN.append((api.version, api.id))\n")})
    load_extensions(cfg)
    assert sys.modules['sdcpp_ext_v'].SEEN == [(1, 'v')]


def test_hooks_and_tabs_are_validated(make_ext):
    cfg = make_ext({
        'bad_hook.py': "def register(api): api.add_hook('on_start', print)\n",
        'bad_tab.py': "def register(api): api.add_ui(print, tabs='upscale')\n",
    })
    assert load_extensions(cfg) == []
    reasons = {e.id: e.reason for e in extensions()}
    assert 'unknown hook' in reasons['bad_hook']
    assert 'unknown tab' in reasons['bad_tab']

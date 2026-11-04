"""A1111-style metadata string: Default sampler / scheduler."""
import pytest

from modules.utils.metadata_utils import (
    build_a1111_metadata, parse_a1111_text
)

PARAMS = {
    'in_pprompt': 'a cat', 'in_nprompt': 'blurry', 'in_steps': 20,
    'in_cfg': 7.0, 'in_width': 512, 'in_height': 512,
    'in_rng': 'Default', 'in_sampler_rng': 'Default',
}


def _sampler(text):
    return text.split('Sampler: ', 1)[1].split(',', 1)[0]


@pytest.mark.parametrize('sampling, scheduler, expected', [
    ('euler_a', 'karras', 'euler_a karras'),
    ('euler_a', 'Default', 'euler_a'),       # no scheduler name to show
    ('Default', 'karras', 'Default karras'),
    ('Default', 'Default', 'Default'),       # not "Default Default"
    ('euler_a', '', 'euler_a'),
])
def test_sampler_field(sampling, scheduler, expected):
    text = build_a1111_metadata(
        {**PARAMS, 'in_sampling': sampling, 'in_scheduler': scheduler}, 42
    )
    assert _sampler(text) == expected


@pytest.mark.parametrize('sampling, scheduler, parsed', [
    ('euler_a', 'karras', ('euler_a', 'karras')),
    ('euler_a', 'Default', ('euler_a', '')),
    ('Default', 'karras', ('Default', 'karras')),
    ('Default', 'Default', ('Default', '')),
])
def test_it_reads_back(sampling, scheduler, parsed):
    text = build_a1111_metadata(
        {**PARAMS, 'in_sampling': sampling, 'in_scheduler': scheduler}, 42
    )
    params = parse_a1111_text(text)
    assert (params['sampler'], params['scheduler']) == parsed


def test_no_stray_space_before_a_comma():
    text = build_a1111_metadata(
        {**PARAMS, 'in_sampling': 'euler_a', 'in_scheduler': 'karras'}, 7
    )
    assert ' ,' not in text
    assert ('RNG: Default, Sampler RNG: Default, '
            'Sampler: euler_a karras') in text

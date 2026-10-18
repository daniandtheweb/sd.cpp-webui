"""Regression tests (C3): a cleared seed must map to -1 (random), not crash.

Background: the seed textbox yields None when cleared (empty -> '').
- CLI:    '-s', str(self._int_param('in_seed'))  -> '-s None' (sd-cli fails
  to parse the seed)
- Server: int(self._get_param('in_seed', -1))    -> TypeError: int(None),
  because _get_param falls back to -1 only when the KEY is missing, not
  when the value is None.

Fix locked in: CommonRunner._get_seed() maps None/'' to -1, keeps 0 as 0,
and is used by both the CLI base args ('-s') and the server payload and
A1111 metadata seed.
"""
import pytest


def _cli_params(seed):
    return {
        'in_pprompt': 'test prompt',
        'in_nprompt': '',
        'in_sampling': 'Euler a',
        'in_steps': 20,
        'in_width': 512,
        'in_height': 512,
        'in_cfg': 7.0,
        'in_seed': seed,
    }


def _cli_flag(command, flag):
    """Returns the value following `flag` in a command list."""
    return command[command.index(flag) + 1]


class TestCliSeed:
    """sd-cli '-s' argument for each seed state."""

    @pytest.mark.parametrize('seed', [None, ''])
    def test_cleared_seed_maps_to_random(self, seed, tmp_path, monkeypatch):
        from modules.core.cli.sdcpp_cli import Txt2ImgRunner
        monkeypatch.chdir(tmp_path)  # keep outputs/ out of the repo
        runner = Txt2ImgRunner('txt2img', _cli_params(seed))
        runner.output_path = str(tmp_path / 'out.png')
        runner._add_base_args()
        assert _cli_flag(runner.command, '-s') == '-1'

    def test_zero_seed_stays_zero(self, tmp_path, monkeypatch):
        from modules.core.cli.sdcpp_cli import Txt2ImgRunner
        monkeypatch.chdir(tmp_path)
        runner = Txt2ImgRunner('txt2img', _cli_params(0))
        runner.output_path = str(tmp_path / 'out.png')
        runner._add_base_args()
        assert _cli_flag(runner.command, '-s') == '0'

    def test_float_seed_normalized(self, tmp_path, monkeypatch):
        from modules.core.cli.sdcpp_cli import Txt2ImgRunner
        monkeypatch.chdir(tmp_path)
        runner = Txt2ImgRunner('txt2img', _cli_params(1234.0))
        runner.output_path = str(tmp_path / 'out.png')
        runner._add_base_args()
        assert _cli_flag(runner.command, '-s') == '1234'


class TestServerSeed:
    """sdapi payload 'seed' for each seed state (previously int(None))."""

    @pytest.mark.parametrize('seed', [None, ''])
    def test_cleared_seed_maps_to_random(self, seed):
        from modules.core.server.sdcpp_server import Txt2ImgApiRunner
        runner = Txt2ImgApiRunner({'in_seed': seed})
        assert runner._build_payload()['seed'] == -1

    def test_zero_seed_stays_zero(self):
        from modules.core.server.sdcpp_server import Txt2ImgApiRunner
        runner = Txt2ImgApiRunner({'in_seed': 0})
        assert runner._build_payload()['seed'] == 0

    def test_float_seed_normalized(self):
        from modules.core.server.sdcpp_server import Txt2ImgApiRunner
        runner = Txt2ImgApiRunner({'in_seed': 1234.0})
        assert runner._build_payload()['seed'] == 1234

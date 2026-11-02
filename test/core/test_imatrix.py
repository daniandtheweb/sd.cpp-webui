"""Imatrix: using imatrices to quantize in the Convert tab."""
import os


def _flags(command, flag):
    """Returns every value following `flag` in a command list."""
    return [command[i + 1] for i, arg in enumerate(command) if arg == flag]


def _convert_command(**params):
    from modules.core.cli.sdcpp_cli import convert
    base = {
        'in_orig_model': 'model.safetensors',
        'in_model_dir': os.path.abspath('models/checkpoints'),
        'in_quant_type': 'q4_K',
    }
    base.update(params)
    return next(convert(base))[0].split(' ')


def test_convert_with_imatrices():
    # Gradio passes the uploaded files' paths
    command = _convert_command(
        in_imatrix_in=['/tmp/gradio/a1/portraits.dat',
                       '/tmp/gradio/b2/landscapes.dat']
    )
    assert [os.path.basename(p) for p in _flags(command, '--imat-in')] == [
        'portraits.dat', 'landscapes.dat'
    ]


def test_convert_without_imatrix():
    assert '--imat-in' not in _convert_command(in_imatrix_in=[])
    assert '--imat-in' not in _convert_command(in_imatrix_in=None)
    assert '--imat-in' not in _convert_command(in_imatrix_in=None)
    assert '--imat-in' not in _convert_command()

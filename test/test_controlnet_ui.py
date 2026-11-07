import gradio as gr

from modules.ui.controlnet import create_cnnet_ui


def _handler(demo, component, event):
    """The function a component's event runs, and its output components."""
    for fn in demo.fns.values():
        if any(block_id == component._id and name == event
               for block_id, name in fn.targets):
            return fn.fn, fn.outputs
    raise AssertionError(f'no {event} handler for {component}')


def _interactive(updates):
    return [u['interactive'] for u in updates]


def _build():
    with gr.Blocks() as demo:
        ui = create_cnnet_ui()
    return demo, ui


def test_widgets_and_keys():
    _, ui = _build()
    assert {'in_cnnet_bool', 'in_cnnet', 'in_control_img',
            'in_control_strength', 'in_cnnet_union',
            'in_control_type'} <= set(ui)
    # the union widgets start switched off, like the rest of ControlNet
    assert ui['in_cnnet_union'].interactive is False
    assert ui['in_control_type'].interactive is False
    assert ui['in_control_type'].choices  # filled from the binary's choices


def test_enabling_controlnet_unlocks_union_and_the_type_only_with_union():
    demo, ui = _build()
    fn, outputs = _handler(demo, ui['in_cnnet_bool'], 'change')
    assert outputs[-1] is ui['in_control_type']
    assert ui['in_cnnet_union'] in outputs

    # enabled, not a union: everything but the control type is usable
    result = _interactive(fn(True, False))
    assert result[:-1] == [True] * (len(result) - 1) and result[-1] is False
    # enabled and a union: the type is usable too
    assert all(_interactive(fn(True, True)))
    # disabled: nothing is, even if union stayed checked
    assert not any(_interactive(fn(False, True)))


def test_the_union_checkbox_unlocks_the_type_only_with_controlnet():
    demo, ui = _build()
    fn, outputs = _handler(demo, ui['in_cnnet_union'], 'change')
    assert outputs == [ui['in_control_type']]
    assert fn(True, True)['interactive'] is True
    assert fn(True, False)['interactive'] is False
    assert fn(False, True)['interactive'] is False


def test_server_model_only_block_hides_the_union_widgets():
    with gr.Blocks():
        ui = create_cnnet_ui(inputs=False)
    assert ui['in_cnnet_union'].visible is False
    assert ui['in_control_type'].visible is False


def test_the_control_image_size_is_shown_under_it(tmp_path):
    from PIL import Image

    demo, ui = _build()
    fn, outputs = _handler(demo, ui['in_control_img'], 'change')
    (info,) = outputs
    assert info.visible is True
    path = tmp_path / 'pose.png'
    Image.new('RGB', (512, 768)).save(path)
    assert fn(str(path)) == 'Control image size: **512 x 768**'
    assert fn(None) == ''                       # cleared
    assert fn(str(tmp_path / 'missing.png')) == ''   # unreadable

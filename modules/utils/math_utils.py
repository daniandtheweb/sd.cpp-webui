"""sd.cpp-webui - utils - math utils module"""

import gradio as gr


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)

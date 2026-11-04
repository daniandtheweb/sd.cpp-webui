"""sd.cpp-webui - utils - math utils module"""

import gradio as gr

_last_seed = None


def random_seed():
    """Sets the seed to -1"""
    return gr.update(value=-1)


def remember_seed(seed):
    """
    Remembers the seed of a generation. A negative seed means sd.cpp picks
    it, so there is nothing to remember until the log reports it.
    """
    global _last_seed
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return
    if seed >= 0:
        _last_seed = seed


def reuse_last_seed():
    """Sets the seed to the one of the last generation, if there was one"""
    if _last_seed is None:
        return gr.skip()
    return gr.update(value=_last_seed)

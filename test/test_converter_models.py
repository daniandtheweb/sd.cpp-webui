from modules.loader import MODEL_DIR_MAP, model_choice
from modules.ui.constants import MODELS


def test_every_converter_model_type_has_a_folder():
    """The Model Type list of the converter must match the folder map."""
    missing = [name for name in MODELS if not MODEL_DIR_MAP.get(name)]
    assert missing == []
    assert model_choice("TAESD")['value'] == MODEL_DIR_MAP["TAESD"]

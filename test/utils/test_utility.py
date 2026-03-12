import os

from modules.utils.file_utils import get_path
from modules.utils.math_utils import random_seed


def test_random_seed_returns_correct_gradio_update_object():
    """
    Tests that the function returns a gr.update() dictionary
    with the value set to -1.
    """
    # Arrange
    # This is what gr.update(value=-1) actually returns:
    expected_output = {'value': -1, '__type__': 'update'}

    # Act
    result = random_seed()

    # Assert
    # We check that the dictionary it returned is exactly
    # what we expect.
    assert result == expected_output


def test_get_path_returns_joined_path_when_filename_provided():
    """Tests the 'happy path' where both arguments are given."""
    # Arrange
    directory = "path/to/data"
    filename = "file.txt"
    # Calculate the expected result exactly how the OS would
    expected = os.path.join(directory, filename)

    # Act
    result = get_path(directory, filename)

    # Assert
    assert result == expected


def test_get_path_returns_none_when_filename_is_none():
    """Tests the case where filename is None."""
    # Arrange
    directory = "path/to/data"
    filename = None

    # Act
    result = get_path(directory, filename)

    # Assert
    assert result is None


def test_get_path_returns_none_when_filename_is_empty_string():
    """
    Tests the edge case where filename is an empty string (which is 'falsy').
    """
    # Arrange
    directory = "path/to/data"
    filename = ""

    # Act
    result = get_path(directory, filename)

    # Assert
    assert result is None

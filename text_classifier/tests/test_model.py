from unittest.mock import patch
from pathlib import Path
from typing import List
import pytest
import torch

from ..model import BoW


@pytest.fixture
def mock_data_file(tmp_path: Path) -> Path:
    """
    Fixture for creating a temporary file.

    Args:
        tmp_path (Path): A temporary directory.

    Returns:
        Path: The path to the created temporary file.
    """
    # Define test data
    test_data = [
        ["tag1", "word1 word2 word3"],
        ["tag2", "word3 word4"],
        ["tag3", "word5 word6"],
    ]

    # Create a test data file
    file_path = tmp_path / "test_file.txt"
    with open(file_path, "wt") as f:
        for line in test_data:
            f.write(" ||| ".join(line) + "\n")

    return file_path


@pytest.fixture
def mock_data() -> List[List[str]]:
    """
    Fixture for providing test data.

    Returns:
        List[List[str]]: A list of lists containing test data.
    """
    test_data = [
        ["tag1", "word1 word2 word3"],
        ["tag2", "word3 word4"],
        ["tag3", "word5 word6"],
    ]
    return test_data


@pytest.mark.model
def test_init():
    """
    Test that the BoW attributes are initialised correctly.
    """
    # Define the number of words and number of tags
    nwords = 100
    ntags = 50
    
    # Create a test model
    bow_model = BoW(nwords, ntags)
    
    assert isinstance(bow_model.Embedding, torch.nn.Embedding)
    assert bow_model.Embedding.weight.size(0) == nwords
    assert bow_model.Embedding.weight.size(1) == ntags


@pytest.mark.model
def test_forward():
    """
    Test that the forward method produces expected results.
    """
    # Define the number of words and number of tags
    nwords = 10
    ntags = 5

    # Create a test model
    bow_model = BoW(nwords, ntags)

    # Create test input data
    x = torch.tensor([1, 2, 3, 4, 5])

    output = bow_model.forward(x)

    assert output.shape == (1, ntags)


@pytest.mark.model
def test_create_dict(mock_data_file):
    """
    Test that the _create_dict class method creates correct WORDTOINDEX and
    TAGTOINDEX dictionaries.
    """
    # Read the contents of the test data file
    with open(mock_data_file, "rt") as f:
        mock_data = [line.strip().split(" ||| ") for line in f]

    BoW._create_dict(mock_data)
    
    BoW._create_dict(mock_data, check_unk=True)
    
    expected_word_to_index = {"<unk>": 0, "word1": 1, "word2": 2, "word3": 3, "word4": 4, "word5": 5, "word6": 6}
    assert BoW.WORDTOINDEX == expected_word_to_index
    assert BoW.TAGTOINDEX == {"tag1": 0, "tag2": 1, "tag3": 2}


@pytest.mark.model
def test_create_tensor(mock_data):
    """
    Test that the _create_tensor class method returns expected tensors.
    """
    # Call the _create_tensor class method with the test data
    tensors = list(BoW._create_tensor(mock_data))
    
    expected_tensors = [
        ([1, 2, 3], 0),
        ([3, 4], 1),
        ([5, 6], 2),
    ]
    assert tensors == expected_tensors


@pytest.mark.model
def test_io_error(tmp_path):
    """
    Test that an IOError exceptio is raised
    when the model_filename argument is invalid.
    """
    # Define the number of words and number of tags
    nwords = 100
    ntags = 10

    # Create a test model
    model = BoW(nwords, ntags)

    # Create a non-existent file path to trigger IOError
    non_existent_file = tmp_path / "non_existent_file.txt"

    with pytest.raises(IOError):
        BoW.load_model(model, non_existent_file)


@pytest.mark.model
def test_generic_exception():
    """
    Test that the load_model static method raises an Exception
    when called with invalid arguments.
    """
    model = ""
    filename = ""

    with pytest.raises(Exception):
        BoW.load_model(model, filename)


@pytest.mark.model
def test_load_model(tmp_path):
    """
    Test that the load_model static method loads model's
    parameters correctly.
    """
    # Define the number of words and number of tags
    nwords = 100
    ntags = 10

    # Create a test model and save its parameters to a file
    model = BoW(nwords, ntags)
    model_filename = tmp_path / "test_model.pth"
    torch.save(model.state_dict(), model_filename)

    # Call the load_model static method with the test inputs
    loaded_model = BoW.load_model(model, model_filename)

    assert isinstance(loaded_model, BoW)

    assert not loaded_model.training


@pytest.mark.model
def test_perform_inference_tag_not_found():
    """
    Test that the perform_inference static method returns "Tag not found".
    """
    # Define test nwords and ntags
    nwords = 10
    ntags = 5

    # Create a test model
    model = BoW(nwords, ntags)

    # Define a test sentence and word_to_index dictionary
    sentence = "This is a test sentence."
    word_to_index = {"This": 0, "is": 1, "a": 2, "test": 3, "sentence.": 4}

    # Define a tag_to_index dictionary that doesn't contain the index for the predicted class
    tag_to_index = {}

    # Call the perform_inference function
    predicted_tag = BoW.perform_inference(model, sentence, word_to_index, tag_to_index, "cpu")

    assert predicted_tag == "Tag not found"


@pytest.mark.model
def test_save_model_exception():
    """
    Test that the save_model static method raises an Exception.
    """
    # Create a test model and model_filename
    model = torch.nn.Linear(10, 2)
    model_filename = "test_model.pth"
    
    # Mock the torch.save function to raise an exception
    with patch('torch.save', side_effect=Exception()), \
        pytest.raises(Exception):
            BoW.save_model(model, model_filename)
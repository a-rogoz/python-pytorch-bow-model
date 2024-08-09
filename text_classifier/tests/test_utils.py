from pathlib import Path
import pytest
import torch

from ..utils import get_number_of_words_and_tags, read_data, sentence_to_tensor


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """
    Fixture for creating a temporary file.

    Args:
        tmp_path (Path): A temporary directory.

    Returns:
        Path: The path to the created temporary file.

    """
    file_path = tmp_path / "test_file.txt"
    with open(file_path, "wt") as f:
        f.write("line1 ||| data1\n")
        f.write("line2 ||| data2\n")
    return file_path


@pytest.mark.utils
def test_get_number_of_words_and_tags():
    """
    Test that the get_number_of_words_and_tags function returns correct values.
    """
    # Set up the word_to_index and tag_to_index dictionaries
    word_to_index = {"word1": 0, "word2": 1, "word3": 2}
    tag_to_index = {"tag1": 0, "tag2": 1}
    
    # Call the get_number_of_words_and_tags function
    number_of_words, number_of_tags = get_number_of_words_and_tags(word_to_index, tag_to_index)
    
    assert number_of_words == 3
    assert number_of_tags == 2


@pytest.mark.utils
def test_read_file_success(temp_file):
    """
    Test that the read_data function reads data correctly.
    """
    data = read_data(temp_file)
    assert data == [["line1", "data1"], ["line2", "data2"]]


@pytest.mark.utils
def test_io_error():
    """
    Test that the read_data function raises the IOError exception
    when given an invalid file path.
    """
    with pytest.raises(IOError):
        read_data("/invalid/file/path.txt")


@pytest.mark.utils
def test_sentence_to_tensor():
    """
    Test that the sentence_to_tensor function converts a sentence
    into tensor correctly.
    """
    # Define a sample sentence and word_to_index mapping
    sentence = "this is a test"
    word_to_index = {"this": 0, "is": 1, "a": 2, "test": 3}

    # Call the sentence_to_tensor function with the sample inputs
    tensor = sentence_to_tensor(sentence, word_to_index)

    assert tensor.shape == (4,)

    expected_indices = [0, 1, 2, 3]
    assert torch.equal(tensor, torch.LongTensor(expected_indices))


@pytest.mark.utils
def test_sentence_to_tensor_exception():
    """
    Test that the sentence_to_tensor function raises the KeyError exception
    when given a sentence with a word not found in the word_to_index dictionary.
    """
    # Define a sentence with a word not found in the word_to_index dictionary
    sentence = "this word is not in dictionary"
    
    # word_to_index dictionary to raise a KeyError
    word_to_index = {"this": 0, "is": 1, "a": 2, "test": 3}

    with pytest.raises(KeyError):
        sentence_to_tensor(sentence, word_to_index) 
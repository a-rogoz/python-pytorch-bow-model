import logging, traceback
from os import strerror
from typing import Dict, List, Tuple
import torch


def get_number_of_words_and_tags(word_to_index: Dict[str, int], tag_to_index: Dict[str, int]) -> Tuple[int, int]:
        """
        Get the number of words and tags.

        Args:
            word_to_index (dict): A dictionary mapping words to their indices.
            tag_to_index (dict): A dictionary mapping tags to their indices.

        Returns:
            Tuple[int, int]: A tuple containing the number of words and the number of tags.
        """
        number_of_words = len(word_to_index)
        number_of_tags = len(tag_to_index)
        return number_of_words, number_of_tags


def read_data(filename: str) -> List[List[str]]:
    """
    Read data from a file.

    Args:
        filename (str): The path to the file to read.

    Returns:
        list: A list containing the data read from the file, where each element represents a line
            split by the ' ||| ' delimiter.

    Raises:
        IOError: If an I/O error occurs while reading the file.
    """
    try:
        Data = []
        with open(filename, 'rt') as f:
            for Line in f:
                Line = Line.lower().strip()
                Line = Line.split(' ||| ')

                Data.append(Line)
        return Data
    except IOError as e:
        logging.error(f"I/O error occurred: {strerror(e.errno)}")
        raise IOError


def sentence_to_tensor(sentence: str, word_to_index: Dict[str, int]) -> torch.LongTensor:
    """
    Convert sentence into tensor using word_to_index dictionary.

    Args:
        sentence (str): The input sentence to convert.
        word_to_index (Dict[str, int]): A dictionary mapping words to their corresponding indices.

    Returns:
        torch.LongTensor: A LongTensor containing the indices of words in the sentence.

    Raises:
        KeyError: If a word in the sentence is not found in the word_to_index dictionary.
    """
    try:
        return torch.LongTensor([word_to_index[_word] for _word in sentence.split(" ")])
    except KeyError as e:
        logging.error("Word not found in the word_to_index dictionary.")
        logging.error(traceback.format_exc())
        raise KeyError
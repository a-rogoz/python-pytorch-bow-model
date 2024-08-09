import logging, random, traceback
from os import strerror
from typing import Dict, Generator, List, Tuple
import torch

from .utils import sentence_to_tensor


class BoW(torch.nn.Module):
    """
    Bag-of-Words model.
    """

    # Word and tag indices
    WORDTOINDEX = {}
    WORDTOINDEX["<unk>"] = len(WORDTOINDEX)

    TAGTOINDEX = {}

    def __init__(self, nwords: int, ntags: int):
        """
        Initialise the Bag-of-Words model.

        Args:
            nwords: The number of words in the vocabulary.
            ntags: The number of tags.
        """
        super(BoW, self).__init__()
        self.Embedding = torch.nn.Embedding(nwords, ntags)
        torch.nn.init.xavier_uniform_(self.Embedding.weight)

        tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.bias = torch.zeros(ntags, requires_grad = True).type(tensor_type)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor containing word indices.

        Returns:
            torch.Tensor: Output tensor representing the model's predictions.
        """
        emb = self.Embedding(x)
        out = torch.sum(emb, dim = 0) + self.bias
        out = out.view(1, -1)
        return out
    

    @classmethod
    def _create_dict(cls, data: List[Tuple[str, str]], check_unk: bool = False) -> None:
        """
        Create dictionaries for words and tags from the given data.

        Args:
            data (list): A list containing tuples of (tag, sentence).
            check_unk (bool, optional): Whether to check for unknown words. Defaults to False.
        """
        for _line in data:
            for word in _line[1].split(" "):
                if check_unk == False:
                    if word not in cls.WORDTOINDEX:
                        cls.WORDTOINDEX[word] = len(cls.WORDTOINDEX)
                else:
                    if word not in cls.WORDTOINDEX:
                        cls.WORDTOINDEX[word] = cls.WORDTOINDEX["<unk>"]

            if _line[0] not in cls.TAGTOINDEX:
                cls.TAGTOINDEX[_line[0]] = len(cls.TAGTOINDEX)

  
    @classmethod
    def _create_tensor(cls, data: List[Tuple[str, str]]) -> Generator[Tuple[List[int], int], None, None]:
        """
        Create a tensor from the given data.

        Args:
            data (list): A list containing tuples of (tag, sentence).

        Yields:
            tuple: A tuple containing a list of word indices and the corresponding tag index.
        """
        for _line in data:
            yield([cls.WORDTOINDEX[word] for word in _line[1].split(" ")], cls.TAGTOINDEX[_line[0]])
    

    @staticmethod
    def load_model(model: torch.nn.Module, model_filename: str) -> torch.nn.Module:
        """
        Load a trained model from a file and return it.

        Args:
            model (torch.nn.Module): The model instance to load the trained parameters into.
            model_filename (str): The filename of the saved model file.

        Returns:
            torch.nn.Module: The loaded model instance with trained parameters.

        Raises:
            IOError: If an I/O error occurs while loading the model from the file.
            Exception: For any other unexpected errors encountered during the loading process.
        """
        try:
            # Load the trained parameters into the model
            model.load_state_dict(torch.load(model_filename))

            # Set the model to evaluation mode
            model.eval()

            return model
        except IOError as e:
            logging.error(f"I/O error occurred: {strerror(e.errno)}")
            raise IOError
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise Exception
        

    @staticmethod
    def perform_inference(model: torch.nn.Module, sentence: str, word_to_index: Dict[str, int], tag_to_index: Dict[str, int], device: str) -> str:
        """
        Perform inference on the trained model.

        Args:
            model (torch.nn.Module): The trained model.
            sentence (str): The input sentence for inference.
            word_to_index (dict): A dictionary mapping words to their indices.
            tag_to_index (dict): A dictionary mapping tags to their indices.
            device (str): "cuda" or "cpu" based on availability.

        Returns:
            str: The predicted class/tag for the input sentence.
        """
        # Preprocess the input sentence to match the model's input format
        sentence_tensor = sentence_to_tensor(sentence, word_to_index)

        # Move the input tensor to the same device as the model
        sentence_tensor = sentence_tensor.to(device)
        
        # Make sure the model is in evaluation mode and on the correct device
        model.eval()
        model.to(device)

        # Perform inference
        with torch.no_grad():
            output = model(sentence_tensor)

        # Move the output tensor to CPU if it's on CUDA
        if device == "cuda":
            output = output.cpu()

        # Convert the model's output to a predicted class/tag
        predicted_class = torch.argmax(output).item()

        # Reverse lookup to get the tag corresponding to the predicted class
        for tag, index in tag_to_index.items():
            if index == predicted_class:
                return tag

        # Return an error message if the tag is not found
        return "Tag not found"
    

    @staticmethod
    def save_model(model: torch.nn.Module, model_filename: str) -> None:
        """
        Save the model parameters to a file.

        Args:
            model: The model to be saved.
            model_filename (str): The filename to save the model state dictionary.

        Raises:
            Exception: If an unexpected error occurs during saving.
        """
        try:
            torch.save(model.state_dict(), model_filename)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            logging.error(traceback.format_exc())
            raise Exception
        

    @staticmethod
    def train_model(model: torch.nn.Module, tensor_type: str, 
                    train_data: List[Tuple[List[int], int]], 
                    test_data: List[Tuple[List[int], int]]) -> str:
        """
        Train and test the model.

        Args:
            model (torch.nn.Module): The model to be trained.
            tensor_type (str): The type of tensor to be used ('cuda' or 'cpu').
            train_data (List[Tuple[List[int], int]]): The training data, where each tuple contains
                a sentence represented as a list of word indices and its corresponding tag.
            test_data (List[Tuple[List[int], int]]): The testing data, where each tuple contains
                a sentence represented as a list of word indices and its corresponding tag.

        Returns:
            str: A log containing the model performance results.
        """

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        for ITER in range(10):
            # Perform training of the model
            model.train()
            random.shuffle(train_data)
            total_loss = 0.0
            train_correct = 0

            for sentence, tag in train_data:
                sentence = torch.tensor(sentence).type(tensor_type)
                tag = torch.tensor([tag]).type(tensor_type)
                output = model(sentence)
                predicted = torch.argmax(output.data.detach()).item()

                loss = criterion(output, tag)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if predicted == tag: train_correct += 1

            # Perform testing of the model
            model.eval()
            test_correct = 0

            for _sentence, tag in test_data:
                _sentence = torch.tensor(_sentence).type(tensor_type)
                output = model(_sentence)
                predicted = torch.argmax(output.data.detach()).item()

                if predicted == tag: test_correct += 1


            # Print model performance results
            log = f'ITER: {ITER+1} | ' \
                f'train loss/sent: {total_loss/len(train_data):.4f} | ' \
                f'train accuracy: {train_correct/len(train_data):.4f} | ' \
                f'test accuracy: {test_correct/len(test_data):.4f}'
            print(log)
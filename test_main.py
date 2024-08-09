from unittest.mock import patch
import pytest
import torch

from main import main, parse_args


@pytest.mark.main
def test_parse_args_without_action():
    """
    Test that the parse_args function raises SystemExit when not
    given a proper action.
    """
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.main
def test_parse_args_with_action_pretest_model_and_sentence():
    """
    Test that the parse_args function returns given arguments.
    """
    with patch('sys.argv', ['main.py', '--action', 'pretest', '--model', 'test_model.pth', '--sentence', 'i love dogs']):
        args = parse_args()
        assert args.action == 'pretest'
        assert args.sentence == 'i love dogs'


@pytest.mark.main
def test_parse_args_with_action_pretest_model_and_without_sentence():
    """
    Test that the parse_args functionn raises SystemExit when not
    given a sample sentence for the pretest action.
    """
    with patch('sys.argv', ['main.py', '--action', 'pretest', '--model', 'test_model.pth']), \
         pytest.raises(SystemExit):
            parse_args()


@pytest.mark.main
def test_parse_args_with_action_infer_model_and_without_sentence():
    """
    Test that the parse_args functionn raises the SystemExit when not
    given a sample sentence for the infer action.
    """
    with patch('sys.argv', ['main.py', '--action', 'infer', '--model', 'test_model.pth']), \
         pytest.raises(SystemExit):
            parse_args()


@pytest.mark.main
def test_main_train_action():
    """
    Test if all functions for the train action are called.
    """
    with patch('sys.argv', ['main.py', '--action', 'train', '--model', 'test_model.pth']), \
        patch('main.read_data') as mock_read_data, \
        patch('main.BoW._create_dict') as mock_create_dict, \
        patch('main.BoW._create_tensor') as mock_create_tensor, \
        patch('main.get_number_of_words_and_tags', return_value=(10, 5)), \
        patch('main.BoW.train_model') as mock_train_model, \
        patch('main.BoW.save_model') as mock_save_model:
            
            main()

            mock_read_data.assert_called_with('data/classes/test.txt')
            mock_create_dict.assert_called()
            mock_create_tensor.assert_called()
            mock_train_model.assert_called()
            mock_save_model.assert_called()


@pytest.mark.main
def test_main_train_action_existing_model_overwrite_no():
    """
    Test that the train action exits when given a filename that exists
    and the user doesn't want to overwrite it.
    """
    with patch('sys.argv', ['main.py', '--action', 'train', '--model', 'test_model.pth']), \
        patch('main.read_data') as mock_read_data, \
        patch('main.BoW._create_dict'), \
        patch('main.BoW._create_tensor'), \
        patch('main.get_number_of_words_and_tags', return_value=(10, 5)), \
        patch('main.BoW.train_model'), \
        patch('main.BoW.save_model'), \
        patch('os.path.exists', return_value=True) as mock_exists, \
        patch('builtins.input', return_value='something_else') as mock_input, \
        patch('builtins.exit') as mock_exit:
    
            main()

            mock_input.assert_called_once_with("Enter 'yes' to overwrite or 'no' to choose a different filename: ")
            mock_exit.assert_called_once_with()


@pytest.mark.main
def test_main_pretest_action():
    """
    Test if all functions for the pretest action are called.
    """
    test_sentence = "I love dogs"

    with patch('sys.argv', ['main.py', '--action', 'pretest', '--model', 'test_model.pth', '--sentence', test_sentence]), \
        patch('main.read_data') as mock_read_data, \
        patch('main.BoW._create_dict') as mock_create_dict, \
        patch('main.BoW._create_tensor') as mock_create_tensor, \
        patch('main.get_number_of_words_and_tags', return_value=(10, 5)), \
        patch('main.sentence_to_tensor', return_value=torch.tensor([1, 2, 3])) as mock_sentence_to_tensor:
        
            main()

            mock_read_data.assert_called_with('data/classes/test.txt')
            mock_create_dict.assert_called()
            mock_create_tensor.assert_called()
            mock_sentence_to_tensor.assert_called()


@pytest.mark.main
def test_main_infer_action():
    """
    Test if all functions for the infer action are called.
    """
    with patch('sys.argv', ['main.py', '--action', 'infer', '--model', 'test_model.pth', '--sentence', 'i love dogs']), \
        patch('main.read_data') as mock_read_data, \
        patch('main.BoW._create_dict') as mock_create_dict, \
        patch('main.BoW._create_tensor') as mock_create_tensor, \
        patch('main.get_number_of_words_and_tags', return_value=(10, 5)), \
        patch('main.BoW.train_model') as mock_train_bow, \
        patch('main.BoW.save_model') as mock_save_model, \
        patch('main.BoW.load_model') as mock_load_model, \
        patch('main.BoW.perform_inference') as mock_perform_inference:

            main()

            mock_read_data.assert_called_with('data/classes/test.txt')
            mock_create_dict.assert_called()
            mock_create_tensor.assert_called()
            mock_train_bow.assert_called()
            mock_save_model.assert_called()
            mock_load_model.assert_called()
            mock_perform_inference.assert_called()


@pytest.mark.main
def test_main_infer_action_unexpected_error():
    """
    Test if the Exception is raised when the perform_inference static method
    encounters unexpected error.
    """
    test_sentence = "I love dogs"

    with patch('sys.argv', ['main.py', '--action', 'infer', '--model', 'test_model.pth', '--sentence', test_sentence]), \
        patch('main.read_data') as mock_read_data, \
        patch('main.BoW._create_dict') as mock_create_dict, \
        patch('main.BoW._create_tensor') as mock_create_tensor, \
        patch('main.get_number_of_words_and_tags', return_value=(10, 5)), \
        patch('main.BoW.train_model') as mock_train_bow, \
        patch('main.BoW.save_model') as mock_save_model, \
        patch('main.BoW.load_model') as mock_load_model, \
        patch('main.BoW.perform_inference', side_effect=Exception()) as mock_perform_inference, \
        pytest.raises(Exception):
            main()
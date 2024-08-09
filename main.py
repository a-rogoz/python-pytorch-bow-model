import argparse, os, logging
import torch

from text_classifier.model import BoW
from text_classifier.utils import (
    get_number_of_words_and_tags,
    read_data,
    sentence_to_tensor
)

# Configure logging
logging.basicConfig(filename="error.log", level=logging.ERROR)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run training, pretesting and inference")
    parser.add_argument("--action", choices=["train", "pretest", "infer"], help="Action to perform", required=True)
    parser.add_argument("--model", help="Model filename", required=True)
    parser.add_argument("--sentence", help="Sample sentence", required=False)
    
    args = parser.parse_args()

    if args.action in ["pretest", "infer"] and args.sentence is None:
        parser.error(f"--sentence is required for action: {args.action}")
    
    return args


def main():
    # Parse provided arguments
    args = parse_args()

    # Trained model filename
    model_filename = args.model

    # Path to the trained model file
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, model_filename)

    # Load training and testing data
    train_data = read_data('data/classes/train.txt')
    test_data = read_data('data/classes/test.txt')

    # Create word to index and tag to index dictionaries from data
    BoW._create_dict(train_data)
    BoW._create_dict(test_data, check_unk=True)

    # Create word and tag tensors from data
    train_data = list(BoW._create_tensor(train_data))
    test_data = list(BoW._create_tensor(test_data))

    # Get the number of words and tags
    number_of_words, number_of_tags = get_number_of_words_and_tags(BoW.WORDTOINDEX, BoW.TAGTOINDEX)

    # Determine the device and the type based on availability
    if torch.cuda.is_available():
        device = "cuda"
        tensor_type = torch.cuda.LongTensor
    else:
        device = "cpu"
        tensor_type = torch.LongTensor

    
    # Check the provided action
    if args.action == "train":
        model = BoW(number_of_words, number_of_tags).to(device)

        # Check if the model was already trained
        if not os.path.exists(file_path):
            # Train the BoW model
            BoW.train_model(model, tensor_type, train_data, test_data)

            # Save the trained BoW model
            BoW.save_model(model, model_filename)

            print("The model was trained and saved to disk.")
        else:
            print(f"Model {model_filename} already exists. Do you want to overwrite it?")
            response = input("Enter 'yes' to overwrite or 'no' to choose a different filename: ")
            if response.lower() == 'yes':
                # Train the BoW model
                BoW.train_model(model, tensor_type, train_data, test_data)

                # Save the trained BoW model
                BoW.save_model(model, model_filename)

                print("The model was retrained and saved to disk.")
            else:
                # Choose a different filename and exit the script
                print("Exiting without retraining the model.")
                exit()
    elif args.action == "pretest":
        # The provided sample sentence
        test_sentence = args.sentence.lower().strip()

        # Pretest the model
        out = sentence_to_tensor(test_sentence, BoW.WORDTOINDEX).type(tensor_type)
        test_model = BoW(number_of_words, number_of_tags).to(device)

        test_model(out)
        print(f"{test_model(out)}")
    elif args.action == "infer":
        # Check if the model was already trained
        if not os.path.exists(file_path):
            model = BoW(number_of_words, number_of_tags).to(device)

            # Train the BoW model
            BoW.train_model(model, tensor_type, train_data, test_data)

            # Save the trained BoW model
            BoW.save_model(model, model_filename)
            
        # Load the trained model
        model = BoW.load_model(BoW(number_of_words, number_of_tags), model_filename)

        # Load word_to_index and tag_to_index dictionaries
        word_to_index, tag_to_index = BoW.WORDTOINDEX, BoW.TAGTOINDEX

        # Perform Inference
        try:
            # The provided sample sentence
            sample_sentence = args.sentence.lower().strip()
            predicted_tag = BoW.perform_inference(model, sample_sentence, word_to_index, tag_to_index, device)
            print(f"Predicted Tag: {predicted_tag}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise Exception
    else:
        print("Invalid action specified.")


if __name__ == "__main__":
    main()
# Bag-of-Words model

## Build a Docker Image
- docker build . -t text-classifier -f Dockerfile

## Run training, pretesting and inference
- docker run --name bow_model text-classifier python main.py --action=train --model="trained_bow_model.pth"
- docker run --name bow_model text-classifier python main.py --action=pretest --model="trained_bow_model.pth" --sentence="i love dogs"
- docker run --name bow_model text-classifier python main.py --action=infer --model="trained_bow_model.pth" --sentence="I love programming"

# Run code locally
- python -m venv .venv
- Activate the virtual environment
- pip install -r requirements.txt

- python main.py --action=train --model="trained_bow_model.pth"
- python main.py --action=pretest --model="trained_bow_model.pth" --sentence="i love dogs"
- python main.py --action=infer --model="trained_bow_model.pth" --sentence="I love programming"

## Run PyTest tests
- pytest -rP

## Check test coverage
- pip install coverage
- coverage run -m pytest
- coverage report
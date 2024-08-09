FROM python:3.12-slim

WORKDIR /

COPY . .

RUN apt-get update && apt-get install -y wget && \
    /usr/local/bin/python -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

RUN wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/dev.txt -O /dev.txt
RUN wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/test.txt -O /test.txt
RUN wget https://raw.githubusercontent.com/neubig/nn4nlp-code/master/data/classes/train.txt -O /train.txt

RUN mkdir data data/classes
RUN cp dev.txt data/classes
RUN cp test.txt data/classes
RUN cp train.txt data/classes
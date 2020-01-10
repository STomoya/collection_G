FROM python:3.7

RUN apt-get update && apt-get upgrade

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /src

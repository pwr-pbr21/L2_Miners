FROM python:3.8-slim

ENV GIT_DISCOVERY_ACROSS_FILESYSTEM=1
RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y git

WORKDIR /github
ADD requirements.txt /github
RUN pip install -r /github/requirements.txt

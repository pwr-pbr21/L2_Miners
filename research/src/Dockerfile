FROM python:3.6-slim

ENV GIT_DISCOVERY_ACROSS_FILESYSTEM=1
RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y git

RUN mkdir /root/technical-roles-script
WORKDIR /root/technical-roles-script
ADD requirements.txt /root/technical-roles-script
RUN pip install -r /root/technical-roles-script/requirements.txt

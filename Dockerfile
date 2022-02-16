FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

## The MAINTAINER instruction sets the Author field of the generated images
MAINTAINER mswzeus@gmail.com
## DO NOT EDIT THESE 3 lines
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt-get etc.

## Do not edit if you have a requirements.txt
RUN pip install -r requirements.txt


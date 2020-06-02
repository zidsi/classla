FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

COPY . /usr/src/stanfordnlp
WORKDIR /usr/src/stanfordnlp

RUN pip install --no-cache-dir protobuf requests tqdm && \
    pip install --no-deps . 

WORKDIR /

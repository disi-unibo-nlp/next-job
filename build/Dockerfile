FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

LABEL maintainer="JobHop"

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
ENV APP_PATH=/workspace

RUN apt-get update -y && \
    apt-get install -y \
        curl \
        git \
        bash \
        nano \
        python3.11 \
        python3-pip \
        python-is-python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed

COPY build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV DEBIAN_FRONTEND=dialog

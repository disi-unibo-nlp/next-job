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
RUN pip install --no-cache-dir \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

COPY build/requirements.txt .
COPY build/constraints.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -c constraints.txt \
    -r requirements.txt
RUN python -c "import torch; assert torch.__version__ == '2.7.1+cu128', torch.__version__; assert torch.version.cuda == '12.8', torch.version.cuda"
RUN python -c "from pylate import indexes, models, retrieve; print('pylate import ok')"

ENV DEBIAN_FRONTEND=dialog

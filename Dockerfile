# Use CUDA runtime image instead of base
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
# Avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive
# Get host user's UID/GID
ARG HOST_UID=1000
ARG HOST_GID=1000
# Install system dependencies and NVIDIA utilities
RUN apt-get update && apt-get install -y \
curl \
 ca-certificates \
 git \
 python3 \
 python3-pip \
 wget \
 libsndfile1-dev \
 tesseract-ocr \
 espeak-ng \
 ffmpeg \
 gosu \
 nvidia-utils-525 \
 && rm -rf /var/lib/apt/lists/*
# Create mluser with host UID/GID
RUN groupadd -g $HOST_GID mluser || groupmod -n mluser $(getent group $HOST_GID | cut -d: -f1) && \
 useradd -u $HOST_UID -g $HOST_GID -d /home/mluser -s /bin/bash -m mluser
# Set up CUDA environment
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
# Switch to mluser
USER mluser
WORKDIR /home/mluser
# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/mluser/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda init bash \
 && . ~/.bashrc \
 && conda install -y python=3.10 \
 && conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 \
 && conda install -y transformers jupyterlab ipython pytest \
 && conda clean -ya \
 && python3 -m pip install --no-cache-dir sentencepiece protobuf safetensors diffusers peft trl
# Create necessary cache directories
RUN mkdir -p /home/mluser/.cache/huggingface /home/mluser/.cache/torch /home/mluser/.cache/pip \
 /home/mluser/.local/bin

RUN python3 -m pip install --no-cache-dir autoawq
# Set ownership of cache directories
USER root
RUN chown -R mluser:mluser /home/mluser/.cache
USER mluser
# Verify CUDA setup
RUN conda run python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN conda run python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
CMD ["bash"]

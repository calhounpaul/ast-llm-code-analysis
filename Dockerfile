ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Get host user's UID/GID
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG PYTHON_VERSION=3.12

# Install system dependencies first (requires root)
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y \
        software-properties-common \
        git \
        curl \
        sudo \
        python3-pip \
        ca-certificates \
        wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Create mluser with host UID/GID
RUN groupadd -g $HOST_GID mluser || groupmod -n mluser $(getent group $HOST_GID | cut -d: -f1) && \
    useradd -u $HOST_UID -g $HOST_GID -d /home/mluser -s /bin/bash -m mluser && \
    echo "mluser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up CUDA environment
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set up Python environment paths early
ENV PATH=/home/mluser/.local/bin:${PATH}
ENV PYTHONPATH=/home/mluser/code_analyzer

# Switch to mluser for remaining operations
USER mluser
WORKDIR /home/mluser

# Create necessary directories
RUN mkdir -p /home/mluser/.cache/huggingface \
    /home/mluser/code_analyzer \
    /home/mluser/target

# Copy requirements file
COPY --chown=mluser:mluser requirements.txt /home/mluser/requirements.txt

# Install Python packages
RUN python3 -m pip install --user --no-cache-dir \
    'cmake>=3.26' \
    ninja \
    packaging \
    'setuptools>=74.1.1' \
    'setuptools-scm>=8' \
    wheel \
    jinja2 \
    toml==0.10.2

RUN python3 -m pip install --user --no-cache-dir -r /home/mluser/requirements.txt

# Copy code analyzer files
COPY --chown=mluser:mluser . /home/mluser/code_analyzer

# Install the code analyzer package in development mode
WORKDIR /home/mluser/code_analyzer
RUN python3 -m pip install --user -e .

WORKDIR /home/mluser

# Default command
CMD ["python3", "-m", "code_analyzer", "/home/mluser/target"]
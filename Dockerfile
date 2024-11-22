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
        ccache \
        software-properties-common \
        git \
        curl \
        sudo \
        vim \
        python3-pip \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libibverbs-dev \
        ca-certificates \
        wget \
        libsndfile1-dev \
        tesseract-ocr \
        espeak-ng \
        gosu \
        cmake \
        ninja-build \
        clang-format \
        libgl1-mesa-glx \
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
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# CUDA arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

# Workaround for Triton issues
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Switch to mluser for remaining operations
USER mluser
WORKDIR /home/mluser

# Create necessary directories
RUN mkdir -p /home/mluser/.cache/huggingface \
    /home/mluser/.cache/torch \
    /home/mluser/.cache/pip \
    /home/mluser/.local/bin

#add /home/mluser/.local/bin to PATH
ENV PATH=/home/mluser/.local/bin:${PATH}

# Install all Python packages as user
# Build dependencies
RUN python3 -m pip install --user --no-cache-dir \
    'cmake>=3.26' \
    ninja \
    packaging \
    'setuptools>=74.1.1' \
    'setuptools-scm>=8' \
    wheel \
    jinja2

# Common dependencies
RUN python3 -m pip install --user --no-cache-dir \
    psutil \
    sentencepiece \
    'numpy<2.0.0' \
    'requests>=2.26.0' \
    tqdm \
    py-cpuinfo \
    'transformers>=4.45.2' \
    'tokenizers>=0.19.1' \
    protobuf \
    'fastapi>=0.107.0,!=0.113.*,!=0.114.0' \
    aiohttp \
    'openai>=1.45.0' \
    'uvicorn[standard]' \
    'pydantic>=2.9' \
    pillow \
    'prometheus_client>=0.18.0' \
    'prometheus-fastapi-instrumentator>=7.0.0' \
    'tiktoken>=0.6.0' \
    'lm-format-enforcer>=0.10.9,<0.11' \
    'outlines>=0.0.43,<0.1' \
    'typing_extensions>=4.10' \
    'filelock>=3.10.4' \
    partial-json-parser \
    pyzmq \
    msgspec \
    'gguf==0.10.0' \
    importlib_metadata \
    'mistral_common[opencv]>=1.5.0' \
    pyyaml \
    'six>=1.16.0' \
    einops \
    'compressed-tensors==0.8.0'

# CUDA-specific dependencies
RUN python3 -m pip install --user --no-cache-dir \
    'ray>=2.9' \
    'nvidia-ml-py>=12.560.30' \
    'torch==2.5.1' \
    'torchvision==0.20.1' \
    'xformers==0.0.28.post3'

# Test dependencies
RUN python3 -m pip install --user --no-cache-dir \
    'absl-py==2.1.0' \
    'accelerate==1.0.1' \
    'aiohappyeyeballs==2.4.3' \
    'aiohttp==3.10.10' \
    'aiosignal==1.3.1' \
    'annotated-types==0.7.0' \
    'anyio==4.6.2.post1' \
    'argcomplete==3.5.1' \
    'async-timeout==4.0.3' \
    'attrs==24.2.0' \
    'audioread==3.0.1' \
    'awscli==1.35.23' \
    'bitsandbytes==0.44.1' \
    'black==24.10.0' \
    'boto3==1.35.57' \
    'buildkite-test-collector==0.1.9' \
    'cupy-cuda12x==13.3.0' \
    'decord==0.6.0' \
    'evaluate==0.4.3' \
    'httpx==0.27.2' \
    'librosa==0.10.2.post1' \
    'matplotlib==3.9.2' \
    'pytest==8.3.3' \
    'pytest-asyncio==0.24.0' \
    'pytest-forked==1.6.0' \
    'pytest-rerunfailures==14.0' \
    'pytest-shard==0.1.2' \
    'ray[adag]==2.35.0' \
    'sentence-transformers==3.2.1' \
    'soundfile==0.12.1' \
    'tensorizer==2.9.0' \
    'timm==1.0.11' \
    'transformers-stream-generator==0.0.5'

# Lint dependencies
RUN python3 -m pip install --user --no-cache-dir \
    'yapf==0.32.0' \
    'toml==0.10.2' \
    'ruff==0.6.5' \
    'codespell==2.3.0' \
    'isort==5.13.2' \
    'clang-format==18.1.5' \
    'sphinx-lint==1.0.0' \
    'mypy==1.11.1' \
    types-PyYAML \
    types-requests \
    types-setuptools

# Install vLLM and FlashInfer as user
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    python3 -m pip install --user --no-cache-dir \
    vllm \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.6/flashinfer-0.1.6+cu121torch2.4-cp${PYTHON_VERSION_STR}-cp${PYTHON_VERSION_STR}-linux_x86_64.whl

RUN python3 -m pip install --user --no-cache-dir \
    'autoawq'

# Set environment variable for vLLM
ENV VLLM_USAGE_SOURCE=production-docker-image
ENV PATH=/home/mluser/.local/bin:${PATH}
ENV PYTHONPATH=/home/mluser/.local/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH}

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" && \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

CMD ["bash"]

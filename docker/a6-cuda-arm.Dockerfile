# CUDA version for Nvidia Grace Hopper
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.11

# CUDA 12.3.1 doesn't provide a cudnn8 image
FROM --platform="linux/arm64" nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 as builder

ARG CUDA_VERSION
ARG PYTHON_VERSION

RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos3.10.2 \
      libgeos-dev \
      # Install opencv via apt to get required libraries
      python3-opencv \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv

RUN python${PYTHON_VERSION} -m ensurepip --upgrade
RUN pip${PYTHON_VERSION} --help

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python${PYTHON_VERSION} -
ENV PATH=/opt/poetry/bin:${PATH}
ENV POETRY_VIRTUALENVS_CREATE=false

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6
COPY docker/ecmwflibs-0.6.1-cp311-cp311-linux_aarch64.whl /opt/

WORKDIR /opt/a6

RUN python${PYTHON_VERSION} -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && poetry add /opt/ecmwflibs-0.6.1-cp311-cp311-linux_aarch64.whl \
 && poetry install --only=main,notebooks


# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM --platform="linux/arm64/v8" nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION

ARG PATH=/venv/bin:${PATH}
ENV PATH=/venv/bin:${PATH}

ENV GIT_PYTHON_REFRESH=quiet

# Copy venv and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/a6 /opt/a6

ARG NCCL_VERSION=2.18.3-1+cuda12.1
# Extract ubuntu distribution version and download the corresponding key.
# This is to fix CI failures caused by the new rotating key mechanism rolled out by Nvidia.
# Refer to https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212771 for more details.
RUN DIST=$(echo ${CUDA_DOCKER_VERSION#*ubuntu} | sed 's/\.//'); \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${DIST}/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu${DIST}/x86_64/7fa2af80.pub \

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends  \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos3.10.2 \
      libgeos-dev \
      libeccodes-tools \
      # Install opencv via apt to get required libraries
      python3-opencv \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      libnccl2=${NCCL_VERSION} \
       libnccl-dev=${NCCL_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Check if all packages successfully installed by importing
RUN which python
RUN python --version
RUN pip list
RUN python -c 'import a6, torch, torchvision, ipykernel, memory_profiler'
RUN python -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'
RUN python -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
RUN python -m cfgrib selfcheck

ENTRYPOINT ["python"]

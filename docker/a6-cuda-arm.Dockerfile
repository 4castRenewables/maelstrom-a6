# NOTE: Building this image requires to remove dependencies:
#   poetry remove torch torchvision ecmwflibs
#
# Then build using:
#
#   docker buildx build --platform="linux/arm64/v8" -t a6-cuda-arm:latest -f docker/a6-cuda-arm.Dockerfile .
#
# Python 3.10 is the native version of the image
ARG PYTHON_VERSION=3.10

# CUDA 12.3.1 doesn't provide a cudnn8 image
FROM --platform="linux/arm64/v8" nvcr.io/nvidia/pytorch:24.01-py3 as builder

ARG PYTHON_VERSION

RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos3.10.2 \
      libgeos-dev \
      libeccodes-tools \
      # opencv breaks hpcx
      # python3-opencv \
      python${PYTHON_VERSION}-dev

# blinker is attempted to be deinstalled when installing the dependencies,
# which fails. Hence, remove it manually via apt.
RUN apt-get remove -y python3-blinker

RUN /opt/nvidia/nvidia_entrypoint.sh

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python${PYTHON_VERSION} -
ENV PATH=/opt/poetry/bin:${PATH}
ENV POETRY_VIRTUALENVS_CREATE=false

# Export to requirements.txt and remove ecmwflibs and torch/torchvision
COPY README.md /opt/a6/
COPY pyproject.toml /opt/a6/
COPY src/a6/ /opt/a6/src/a6
# Rename wheel to cp310 to avoid error from pip
COPY docker/ecmwflibs-0.6.1-cp311-cp311-linux_aarch64.whl /opt/ecmwflibs-0.6.1-cp310-cp310-linux_aarch64.whl

WORKDIR /opt/a6

RUN pip install --upgrade pip
RUN poetry add -vvv /opt/ecmwflibs-0.6.1-cp310-cp310-linux_aarch64.whl
RUN poetry install --only=main,notebooks

# Open CV import fails and has to be fixed
# 1. download the autofix tool
RUN pip install opencv-fixer==0.2.5
# 2. execute
RUN python -c "from opencv_fixer import AutoFix; AutoFix()"

# Cleanup
RUN apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN rm /opt/ecmwflibs-0.6.1-cp310-cp310-linux_aarch64.whl
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

ENV GIT_PYTHON_REFRESH=quiet

# Check if all packages successfully installed by importing
RUN which python
RUN python --version
RUN pip list
RUN python -c 'import a6, torch, torchvision, ipykernel, memory_profiler'
RUN python -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
RUN python -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'
RUN python -m cfgrib selfcheck
RUN python -m eccodes selfcheck

ENTRYPOINT ["python"]

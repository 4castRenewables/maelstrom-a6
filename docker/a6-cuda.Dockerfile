# CUDA version for Nvidia A2
#ARG CUDA_VERSION=11.8.0
# Default CUDA version
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 as builder

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

RUN python${PYTHON_VERSION} -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6

WORKDIR /opt/a6

RUN . /venv/bin/activate \
 && poetry export -f requirements.txt --output requirements.txt \
 && pip install -r requirements.txt \
 # Below code for updating torch is only required for CUDA 11.7
 #&& poetry source add --priority=supplemental pytorch-cuda118 https://download.pytorch.org/whl/cu118 \
 #&& poetry add \
 #   -vvv \
 #   --source pytorch-cuda118 \
 #   torch==$(poetry show torch | awk '/version/ { print $3 }') \
 #   torchvision==$(poetry show torchvision | awk '/version/ { print $3 }') \
 && poetry install -vvv --only=main,notebooks


# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ARG PYTHON_VERSION

ARG PATH=/venv/bin:${PATH}
ENV PATH=/venv/bin:${PATH}

ENV GIT_PYTHON_REFRESH=quiet

# Copy venv and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/a6 /opt/a6

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends  \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos3.10.2 \
      libgeos-dev \
      # Install opencv via apt to get required libraries
      python3-opencv \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
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
RUN python -m eccodes selfcheck

ENTRYPOINT ["python"]

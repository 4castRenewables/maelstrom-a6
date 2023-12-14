# NCCL is not correctly installed when using CUDA >=11.7.1
ARG CUDA_VERSION=12.1.0

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04 as builder

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.11

RUN apt-get update \
 && apt-get install -y software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget \
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
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3.11 -
ENV PATH=/opt/poetry/bin:${PATH}
ENV POETRY_VIRTUALENVS_CREATE=false

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6

# Must install a6 after unpacking since conda doesn't allow to pack
# packages installed in editable mode.
RUN python3.11 -m venv /venv
WORKDIR /opt/a6
RUN . /venv/bin/activate \
 && poetry add torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
 && poetry install --only=main,notebooks

RUN . /venv/bin/activate \
 && python -c 'import a6, torch, torchvision, ipykernel, memory_profiler' \
 && python -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'


ENV export TORCH_CUDA_ARCH_LIST="8.0"
# Install apex
RUN git clone https://github.com/NVIDIA/apex /opt/apex \
 && cd /opt/apex \
 && . /venv/bin/activate \
 && pip install packaging \
 && pip install -v  \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-build-isolation \
      --config-settings "--build-option=--cpp_ext" \
      --config-settings "--build-option=--cuda_ext" \
      ./

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04

ARG PATH=/venv/bin:${PATH}
ENV PATH=/venv/bin:${PATH}

ENV GIT_PYTHON_REFRESH=quiet

# VISSL conda env and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/a6 /opt/a6

ARG PYTHON_VERSION=3.11

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


RUN which python
RUN python --version
RUN pip list
RUN python -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
RUN python -c 'import a6, torch, torchvision, ipykernel, memory_profiler'
RUN python -c 'import apex'
RUN python -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'
RUN python -m cfgrib selfcheck

ENTRYPOINT ["python"]

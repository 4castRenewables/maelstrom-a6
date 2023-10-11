# This image is just for testing the installation.
# It builds much faster than the Apptainer image due to caching.

ARG CUDA_VERSION=11.7.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04 as builder

SHELL ["/bin/bash", "-c"]

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
 && apt-key del 3bf863cc \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget \
      curl \
      git \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos-3.8.0 \
      libgeos-dev

# Install conda (miniconda)
RUN wget \
      --quiet \
      -O miniconda.sh \
      https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.11
# torch 2.1.0 requries CUDA 11.8.0 and has no
# compatible apex version on conda, hence use 2.0.1
ARG PYTORCH_VERSION=2.0.1
ARG TORCHVISION_VERSION=0.15.2
ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}

ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create and activate conda environment
RUN conda config --add channels conda-forge \
 && conda install conda-pack \
 && conda update pip setuptools \
 && conda create --name a6 python=${PYTHON_VERSION}

# Install PyTorch and apex
RUN conda install -n a6 -c pytorch pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION}
RUN conda install -n a6 -c anaconda cudatoolkit=${CUDA_VERSION}
RUN conda install -n a6 -c conda-forge nvidia-apex

# Use conda-pack to create a standalone env in /venv and install vissl
RUN conda-pack -n a6 -o /opt/env.tar.gz \
 && mkdir /venv \
 && tar -xzf /opt/env.tar.gz -C /venv \
 && . /venv/bin/activate \
 && conda-unpack

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python -
ENV PATH=/opt/poetry/bin:${PATH}

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6

# Must install a6 after unpacking since conda doesn't allow to pack
# packages installed in editable mode.
WORKDIR /opt/a6
RUN . /venv/bin/activate \
 && POETRY_VIRTUALENVS_CREATE=false poetry install --only=main

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu20.04

ARG CUDA_VERSION

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}

ENV GIT_PYTHON_REFRESH=quiet

# VISSL conda env and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/a6 /opt/a6

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
 && apt-key del 3bf863cc \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install opencv via apt to get required libraries
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends  \
      # Installs gcc, required by HDBSCAN
      build-essential \
      # Required by cartopy
      libgeos-3.8.0 \
      libgeos-dev \
      python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN which python \
 && python --version \
 && pip list \
 && python -c 'import a6, apex, torch, torchvision' \
 && python -m cfgrib selfcheck

ENTRYPOINT ["python"]

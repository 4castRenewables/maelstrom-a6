# This image is just for testing the installation.
# It builds much faster than the Apptainer image due to caching.

ARG CUDA_VERSION=11.1.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu18.04 as builder

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG VISSL_VERSION=0.1.6

SHELL ["/bin/bash", "-c"]

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
      git

# Install conda (miniconda)
RUN wget \
      --quiet \
      -O miniconda.sh \
      https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda

# Create and activate conda environment
RUN conda config --add channels conda-forge \
 && conda install conda-pack \
 && conda update pip setuptools \
 && conda create --name vissl python=${PYTHON_VERSION}

# Install PyTorch and apex
RUN conda install -n vissl \
        -c pytorch -c conda-forge \
        pytorch=${PYTORCH_VERSION} \
        torchvision=${TORCHVISION_VERSION} \
        cudatoolkit=${CUDA_VERSION} \
 && conda install -n vissl -c vissl apex

# Use conda-pack to create a standalone env in /venv
RUN conda-pack -n vissl -o /opt/env.tar.gz \
 && mkdir /venv \
 && tar -xzf /opt/env.tar.gz -C /venv \
 && . /venv/bin/activate \
 && conda-unpack

# Install VISSL
# Clone VISSL repository and install
RUN git clone --recursive https://github.com/facebookresearch/vissl.git /opt/vissl
WORKDIR /opt/vissl
RUN git checkout v${VISSL_VERSION} \
 && git checkout -b v${VISSL_VERSION}

# Install VISSL dependencies and VISSL in dev mode
# Update classy vision install to commit stable for vissl.
# Update fairscale install to commit stable for vissl.
# Update numpy to fix ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
# See https://stackoverflow.com/a/74934626/16814206
# Note: If building from vissl main, use classyvision main.
RUN . /venv/bin/activate \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir opencv-python \
 && pip install --no-cache-dir -e ".[dev]" \
 && pip uninstall -y \
      classy_vision \
      fairscale \
      numpy \
 && pip install --no-cache-dir \
      classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d \
      fairscale==0.4.6 \
      numpy==1.21.0

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu18.04

ARG CUDA_VERSION

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}

# VISSL conda env and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/vissl /opt/vissl

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
 && apt-key del 3bf863cc \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install opencv via apt to get required libraries
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 && apt-get install -y --no-install-recommends python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN which python \
 && python --version \
 && python -c 'import torch, apex, vissl, cv2'

ENTRYPOINT ["python"]
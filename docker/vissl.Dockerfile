ARG CUDA_VERSION=10.2

FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel as builder

ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION
ARG PYTORCH_VERSION=1.8.1
ARG TORCHVISION_VERSION=0.9.1
ARG VISSL_VERSION=0.1.6

SHELL ["/bin/bash", "-c"]

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 && \
    apt-key del 3bf863cc && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget \
      git

# Install conda (miniconda)
RUN wget \
      --quiet \
      -O miniconda.sh \
      https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda

# Create and activate conda environment
RUN conda config --add channels conda-forge && \
    conda update pip setuptools && \
    conda create --name vissl python=${PYTHON_VERSION}

# Install PyTorch and apex
RUN conda install -n vissl \
        -c pytorch -c conda-forge \
        pytorch=${PYTORCH_VERSION} \
        torchvision=${TORCHVISION_VERSION} \
        cudatoolkit=${CUDA_VERSION} && \
    conda install -n vissl \
        -c vissl -c iopath -c conda-forge -c pytorch -c defaults \
        apex

# Install VISSL
# Clone VISSL repository and install
RUN git clone --recursive https://github.com/facebookresearch/vissl.git /opt/vissl
WORKDIR /opt/vissl
RUN git checkout v${VISSL_VERSION}

# Set conda shell
SHELL ["conda", "run", "-n", "vissl", "/bin/bash", "-c"]

# Install VISSL dependencies and VISSL in dev mode
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir opencv-python && \
    python -m pip install --no-cache-dir -e "."

# Update classy vision install to commit stable for vissl.
# Update fairscale install to commit stable for vissl.
# Update numpy to fix ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
# See https://stackoverflow.com/a/74934626/16814206
# Note: If building from vissl main, use classyvision main.
RUN python -m pip uninstall -y \
      classy_vision \
      fairscale \
      numpy
RUN python -m pip install --no-cache-dir \
      classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d \
      fairscale==0.4.6 \
      numpy==1.21.0

# Delete Python cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf \
 && cd /opt/conda/envs/vissl \
 && find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM nvidia/cuda:${CUDA_VERSION}-cudnn7-devel

ARG CUDA_VERSION

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}

# VISSL conda env and repo
COPY --from=builder /opt/conda/envs/vissl /venv
COPY --from=builder /opt/vissl /opt/vissl

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 && \
    apt-key del 3bf863cc && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install opencv via apt to get required libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sv /venv/bin/python /usr/bin/python

## Check correct installation
#RUN . /opt/conda/etc/profile.d/conda.sh && \
#    conda activate vissl && \
#    which python && \
#    python --version && \
#    python -c 'import torch, apex, vissl, cv2' \

RUN which python && \
    python --version && \
    python -c 'import torch, apex, vissl, cv2'

ENTRYPOINT ["python"]

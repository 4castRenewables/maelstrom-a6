ARG PYTHON_VERSION=3.11

# NOTE: When updating ROCM version, make sure to change it
# for both layers as well ass the `poetry add source` command.
FROM rocm/dev-ubuntu-22.04:5.6-complete as builder

ARG ROCM_VERSION
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

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6

WORKDIR /opt/a6

RUN python${PYTHON_VERSION} -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && poetry source add --priority=supplemental pytorch-rocm https://download.pytorch.org/whl/rocm5.6 \
 && poetry add \
    -vvv \
    --source pytorch-rocm \
    torch==$(poetry show torch | awk '/version/ { print $3 }') \
    torchvision==$(poetry show torchvision | awk '/version/ { print $3 }') \
 && poetry install --only=main,notebooks

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM rocm/dev-ubuntu-22.04:5.6-complete

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

# Inlcude ROCm SMI libraries for python
ENV PYTHONPATH=/opt/rocm/libexec/rocm_smi/:$PYTHONPATH

# Check if all packages successfully installed by importing
RUN which python
RUN python --version
RUN pip list
RUN python -c 'import rsmiBindings'
RUN python -c 'import a6, torch, torchvision, ipykernel, memory_profiler'
RUN python -c 'import torch.distributed.distributed_c10d as c10d; assert c10d._NCCL_AVAILABLE, "NCCL not available"'
RUN python -c 'from torch._C._distributed_c10d import ProcessGroupNCCL'
RUN python -m cfgrib selfcheck

ENTRYPOINT ["python"]

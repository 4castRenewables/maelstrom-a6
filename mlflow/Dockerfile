ARG PYTHON_VERSION="3.10"

FROM python:${PYTHON_VERSION}-slim-bullseye as builder

ADD dist/ /opt/dist/

WORKDIR /opt

RUN apt-get update -y \
 && apt-get install -y \
    # Installs gcc, required by HDBSCAN
    build-essential \
    # Required by cartopy
    libgeos-3.9.0 \
    libgeos-dev

# Install Python package
RUN python -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && pip install /opt/dist/*.whl

# Delete Python cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf \
 && cd /venv \
 && find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM python:${PYTHON_VERSION}-slim

COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

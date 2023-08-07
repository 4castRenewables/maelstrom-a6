ARG PYTHON_VERSION="3.10"

FROM fabianemmi/python-poetry:${PYTHON_VERSION}-1.5.1-slim-bullseye as builder

COPY README.md/ /opt/a6/
COPY pyproject.toml /opt/a6/
COPY poetry.lock /opt/a6/
COPY src/a6/ /opt/a6/src/a6

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/a6

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
 && POETRY_VIRTUALENVS_CREATE=false poetry install --only=main

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM python:${PYTHON_VERSION}-slim-bullseye

RUN apt-get update -y \
 && apt-get install -y \
    # Required by cartopy
    libgeos-3.9.0 \
    libgeos-dev \
    # Required by opencv
    python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
COPY --from=builder /opt/a6 /opt/a6

ENV PATH=/venv/bin:$PATH
ENV GIT_PYTHON_REFRESH=quiet

RUN which python \
 && python --version \
 && pip list \
 && python -c 'import a6'

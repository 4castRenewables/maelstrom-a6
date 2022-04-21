ARG PYTHON_VERSION="3.9"
ARG POETRY_VERSION="1.1.13"

FROM python:${PYTHON_VERSION}-slim-buster as builder

# Install poetry for Python.
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=$POETRY_VERSION \
  POETRY_HOME=/opt/poetry \
  POETRY_VIRTUALENVS_CREATE=false

ENV PATH="${POETRY_HOME}/bin:${PATH}"

RUN apt-get update && apt-get install -y curl && \
  curl -sSL https://install.python-poetry.org | python

# Install the local package.
ENV CODE_DIR=/opt/code

ADD src/lifetimes ${CODE_DIR}/src/lifetimes
ADD poetry.lock ${CODE_DIR}
ADD pyproject.toml ${CODE_DIR}

WORKDIR /opt/code

# Install with poetry
RUN python -m venv /venv && \
  /venv/bin/python -m pip install --upgrade pip && \
  . /venv/bin/activate && \
  poetry install --no-dev --no-root --no-ansi && \
  poetry build && \
  pip install ${CODE_DIR}/dist/*.whl

# Killing Python Cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf && \
  cd /venv && \
  find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM python:${PYTHON_VERSION}-slim-buster

# Add venv
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

# Add the data
ADD data /data

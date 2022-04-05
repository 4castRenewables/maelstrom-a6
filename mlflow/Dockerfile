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
  curl -sSL https://install.python-poetry.org | python - && \
  python -m venv /venv

# Install the local package.
ENV CODE_DIR=/opt/code

ADD src/lifetimes ${CODE_DIR}/src/lifetimes
ADD poetry.lock ${CODE_DIR}
ADD pyproject.toml ${CODE_DIR}

WORKDIR /opt/code
RUN . /venv/bin/activate && poetry install --no-dev

# Killing Python Cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf && \
  cd /venv && \
  find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf



FROM python:${PYTHON_VERSION}-slim-buster

COPY --from=builder /venv /venv

# Note (fe): MLflow mounts the local directory and overwrites the entrypoint when running the image with Docker.
#            Thus, we likely don't have to include any source files or create an entrypoint.
# ENV CODE_DIR=/opt/code
#
# ADD data/ /opt/data
# ADD mlflow/main.py ${CODE_DIR}/main.py
#
RUN echo ". /venv/bin/activate && python" > entrypoint.sh
#
#WORKDIR /opt/code
#
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

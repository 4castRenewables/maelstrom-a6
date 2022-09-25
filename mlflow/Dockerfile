ARG PYTHON_VERSION="3.9"

FROM python:${PYTHON_VERSION}-slim-buster as builder

ADD dist/ /opt/

WORKDIR /opt

# Install gcc, which is required for HDBSCAN
RUN apt-get update -y \
 && apt-get install -y build-essential

# Install Python package
RUN python -m venv /venv \
 && /venv/bin/python -m pip install --upgrade pip \
 && . /venv/bin/activate \
 && pip install *.whl

# Delete Python cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf \
 && cd /venv \
 && find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM python:${PYTHON_VERSION}-slim-buster

# Add venv
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

ARG PYTHON_VERSION="3.9"

FROM python:${PYTHON_VERSION}-slim-buster as builder

ADD dist/ /opt/dist/

WORKDIR /opt

# Install gcc, which is required for HDBSCAN
RUN apt-get update -y \
 && apt-get install -y build-essential

# Install Python package
RUN python -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && pip install /opt/dist/*.whl \
 # HDBSCAN import fails for joblib>1.1.0
 && pip install joblib==1.1.0

# Delete Python cache
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf \
 && cd /venv \
 && find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM python:${PYTHON_VERSION}-slim-buster

# Add venv
COPY --from=builder /venv /venv
ENV PATH=/venv/bin:$PATH

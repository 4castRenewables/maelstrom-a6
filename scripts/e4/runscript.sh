#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SINGULARITY_IMAGE_PATH=/home/femmerich/.singularity/jupyter-kernel.sif

module load go-1.17.6/singularity-3.9.5

singularity exec \
  --cleanenv \
  -B /data:/data \
  --env SINGULARITY_IMAGE=${SINGULARITY_IMAGE_PATH} \
  ${SINGULARITY_IMAGE_PATH} \
  python \
  ${SCRIPT_DIR}/application.py

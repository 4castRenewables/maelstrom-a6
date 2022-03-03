#!/bin/bash

export SSH_PRIVATE_KEY_FILE=e4
export E4_USER=femmerich
export LOGIN_MACHINE=172.18.19.216

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

scp -i ~/.ssh/${SSH_PRIVATE_KEY_FILE} $SCRIPT_DIR/jupyter_kernel.sif  ${E4_USER}@${LOGIN_MACHINE}:/home/${E4_USER}/.singularity/jupyter-kernel.sif

scp -i ~/.ssh/${SSH_PRIVATE_KEY_FILE} $SCRIPT_DIR/kernel.json  ${E4_USER}@${LOGIN_MACHINE}:/home/${E4_USER}/.local/share/jupyter/kernels/lifetimes/kernel.json

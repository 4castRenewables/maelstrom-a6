#!/bin/bash

export SSH_PRIVATE_KEY_FILE=fzj-juwels
export JUDOOR_USER=emmerich1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

scp -i ~/.ssh/${SSH_PRIVATE_KEY_FILE} $SCRIPT_DIR/jupyter_kernel.sif ${JUDOOR_USER}@juwels-cluster.fz-juelich.de:/p/scratch/deepacf/${JUDOOR_USER}/jupyter-lifetimes/jupyter-kernel.sif

scp -i ~/.ssh/${SSH_PRIVATE_KEY_FILE} $SCRIPT_DIR/kernel.json ${JUDOOR_USER}@juwels-cluster.fz-juelich.de:/p/home/jusers/${JUDOOR_USER}/juwels/.local/share/jupyter/kernels/lifetimes/

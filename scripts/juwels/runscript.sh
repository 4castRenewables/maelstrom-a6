#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

singularity exec \
  --cleanenv \
  -B /usr/:/host/usr/,/etc/slurm:/etc/slurm,/usr/lib64:/host/usr/lib64,/opt/parastation:/opt/parastation,/usr/lib64/slurm:/usr/lib64/slurm,/usr/share/lua:/usr/share/lua,/usr/lib64/lua:/usr/lib64/lua,/opt/jsc:/opt/jsc,/var/run/munge:/var/run/munge \
  /p/scratch1/deepacf/emmerich1/jupyter-a6/jupyter-kernel.sif \
  python \
  ${SCRIPT_DIR}/application.py

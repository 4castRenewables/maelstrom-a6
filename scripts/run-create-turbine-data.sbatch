#!/bin/bash
#SBATCH -p batch
#SBATCH -A deepacf
#SBATCH -o job.out
#SBATCH -e job.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1

srun apptainer run -B /p/home/jusers/emmerich1/juwels/code/a6:/opt/a6 /p/project/deepacf/emmerich1/a6-cuda.sif python /opt/a6/scripts/create_turbine_data.py

#!/bin/bash
#SBATCH -A exalab
#SBATCH -p dc-cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48


export SRUN_CPUS_PER_TASK=48

srun apptainer run \
    -B /p/home/jusers/emmerich1/juwels/code/a6:/opt/a6 \
    /p/project/hclimrep/emmerich1/a6-cuda.sif \
    python /opt/a6/mlflow/forecast_example.py \
      --turbine-id 4 \
      --no-testing
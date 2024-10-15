#!/bin/bash
#SBATCH -A deepacf
#SBATCH -p batch
#SBATCH --time=01:10:00
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12

# Duration: roughly 1 hour

export SRUN_CPUS_PER_TASK=12

srun apptainer run \
    -B ${HOME}/code/a6:/opt/a6 \
    /p/project1/deepacf/emmerich1/a6-cuda.sif \
    python /opt/a6/mlflow/forecast_errors.py \
      --preprocessed-data-dir /p/home/jusers/emmerich1/juwels/data/production-preprocessed \
      --pca-path /p/project1/deepacf/emmerich1/data/pca_kmeans_lswrs_30_40.nc \
      --pca-n-clusters 30 \
      --gwl-path /opt/a6/src/tests/data/gwl.nc \
      --dcv2-path /p/project1/deepacf/emmerich1/data/dcv2-lswrs.nc \
      --random \
      --train-size 365 \
      --results-dir /p/project/deepacf/emmerich1/data/forecast-errors \
      --parallel
#!/bin/bash
#SBATCH -A hclimrep
#SBATCH -p booster
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1


export SRUN_CPUS_PER_TASK=48

srun apptainer run \
    --cleanenv \
    --env OPENBLAS_NUM_THREADS=1 \
    --env TESTING=${TESTING:-false} \
    -B /p/home/jusers/$USER/juwels/code/a6:/opt/a6 \
    /p/project/hclimrep/$USER/a6-cuda.sif \
    python /opt/a6/mlflow/pca_k_means_scree_test.py
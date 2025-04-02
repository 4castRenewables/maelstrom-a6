#!/bin/bash
#SBATCH -A exalab
#SBATCH -p dc-cpu-bigmem
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128


export SRUN_CPUS_PER_TASK=128

srun apptainer run \
    --cleanenv \
    --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OMP_NUM_THREADS=1 \
    --env N_PCS_PCA_START=20 \
    -B /p/home/jusers/$USER/juwels/code/a6:/opt/a6 \
    /p/project/hclimrep/$USER/a6-cuda.sif \
    python /opt/a6/scripts/pca_k_means_scree_test.py
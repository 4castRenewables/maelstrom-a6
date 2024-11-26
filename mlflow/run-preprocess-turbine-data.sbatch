#!/bin/bash
#SBATCH -A exalab
#SBATCH -p dc-cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=45
#SBATCH --cpus-per-task=1

# Duration: roughly 10 minutes

export SRUN_CPUS_PER_TASK=1

srun apptainer run \
    -B /p/home/jusers/emmerich1/juwels/code/a6:/opt/a6 \
    /p/project/hclimrep/emmerich1/a6-cuda.sif \
    python /opt/a6/mlflow/preprocess_turbine.py \
      --pressure-level-data /p/project/hclimrep/emmerich1/data/ecmwf_ifs/pl_2017_2020.nc \
      --model-level-data /p/project/hclimrep/emmerich1/data/ecmwf_ifs/ml_2017_2020.nc \
      --surface-level-data /p/project/hclimrep/emmerich1/data/ecmwf_ifs/sfc_2017_2020.nc \
      --turbine-data-dir /p/home/jusers/emmerich1/juwels/data/production \
      --output-dir /p/home/jusers/emmerich1/juwels/data/production-preprocessed

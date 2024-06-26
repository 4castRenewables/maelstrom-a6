#!/bin/bash -l

#BATCH -p batch           #batch partition
#SBATCH -J ipy_engines      #job name
#SBATCH -N 2                # 2 node, you can increase it
#SBATCH -n 10                # 10 task, you can increase it
#SBATCH -c 1                # 1 cpu per task
#SBATCH -t 1:00:00         # Job is killed after 1h

SINGULARITY_IMAGE=/p/scratch1/deepacf/${USER}/jupyter-a6/jupyter-kernel.sif

#create a new ipython profile appended with the job id number
PROFILE=job_${SLURM_JOB_ID}

echo "Loading Jupyter module"
module load Jupyter/2021.3.2-Python-3.8.5
echo "Creating profile_${PROFILE}"
ipython profile create ${PROFILE}

# Number of tasks - 1 controller task - 1 python task
export N_WORKERS=$((${SLURM_NTASKS}-2))

LOG_DIR="$(pwd)/logs/job_${SLURM_JOBID}"
mkdir -p ${LOG_DIR}

#srun: runs ipcontroller -- forces to start on first node
srun \
  --output=${LOG_DIR}/ipcontroller-%j-workers.out \
  --exclusive \
  -N 1 \
  -n 1 \
  -c ${SLURM_CPUS_PER_TASK} \
  singularity \
  exec \
  --cleanenv \
  ${SINGULARITY_IMAGE} \
  ipcontroller \
  --ip="*" \
  --profile=${PROFILE} &
sleep 10

#srun: runs ipengine on each available core -- controller location first node
srun \
  --output=${LOG_DIR}/ipengine-%j-workers.out \
  --exclusive \
  -n ${N_WORKERS} \
  -c ${SLURM_CPUS_PER_TASK} \
  singularity \
  exec \
  --cleanenv \
  ${SINGULARITY_IMAGE} \
  ipengine \
  --profile=${PROFILE} \
  --location=$(hostname) &
sleep 25

#srun: starts job
echo "Launching job for script $1"
srun \
  --output=${LOG_DIR}/code-%j-execution.out  \
  --exclusive \
  -N 1 \
  -n 1 \
  -c ${SLURM_CPUS_PER_TASK} \
  singularity \
  exec \
  --cleanenv \
  ${SINGULARITY_IMAGE} \
  python $1 \
  -p ${PROFILE}

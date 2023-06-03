#!/bin/bash -x
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --output=/p/home/jusers/emmerich1/juwels/job-logs/%j.out
#SBATCH --error=/p/home/jusers/emmerich1/juwels/job-logs/%j.err

NODES=${NODES:-1}
N_GPUS=${N_GPUS:-4}
EPOCHS=${EPOCHS:-1}
DATASET=${DATASET:-"[remote_dataset_jsc_hourly]"}

if [ "$NODES" -gt "1" ]; then
    RUN_ID=${SLURMD_NODENAME}i:29500;
else
    RUN_ID=auto;
fi

if [ "$EPOCHS" -gt "999" ]; then
    LENGTHS="[0.01,0.99]";
elif [ "$EPOCHS" -gt "799" ]; then
    LENGTHS="[0.0125,0.9875]";
elif [ "$EPOCHS" -gt "599" ]; then
    LENGTHS="[0.0166667,0.9833333]";
elif [ "$EPOCHS" -gt "499" ]; then
    LENGTHS="[0.02,0.98]";
elif [ "$EPOCHS" -gt "399" ]; then
    LENGTHS="[0.025,0.975]";
elif [ "$EPOCHS" -gt "101" ]; then
    LENGTHS="[0.05,0.95]";
elif [ "$EPOCHS" -gt "5" ]; then
    LENGTHS="[0.1,0.9]";
elif [ "$EPOCHS" -gt "2" ]; then
    LENGTHS="[0.002563,0.997437]";
elif [ "$EPOCHS" -gt "1" ]; then
    LENGTHS="[0.00641,0.99359]";
else
    LENGTHS="[0.0128,0.9872]";
fi

echo "Submitting VISSL"
echo "NODES=${NODES}"
echo "N_GPUS=${N_GPUS}"
echo "EPOCHS=${EPOCHS}"
echo "LENGTHS=${LENGTHS}"
echo "RUN_ID=${RUN_ID}"
echo "DATASET=${DATASET}"

REPO_DIR=/p/home/jusers/emmerich1/juwels/code/a6/mlflow/deepclusterv2

# BASE_DIR=/p/scratch/deepacf/maelstrom/emmerich1/deepclusterv2/${SLURM_JOB_ID}
BASE_DIR=/p/scratch/deepacf/maelstrom/emmerich1/deepclusterv2/real
CHECKPOINTS_DIR=${BASE_DIR}/checkpoints
TENSORBOARD_LOG_DIR=${BASE_DIR}/tensorboard-logs
TMP_DATA_DIR=${BASE_DIR}/tmp
CLUSTERING_DIR=${BASE_DIR}/clustering

mkdir -p ${BASE_DIR} ${CHECKPOINTS_DIR} ${TENSORBOARD_LOG_DIR} ${TMP_DATA_DIR}

echo "Run directory created at ${BASE_DIR}"

source /p/project/deepacf/maelstrom/emmerich1/venvs/mantik/bin/activate
source /p/home/jusers/emmerich1/juwels/code/a6/mantik.env > /dev/null 2>&1
eval $(mantik init)
unset MANTIK_USERNAME
unset MANTIK_PASSWORD
echo "MLFLOW_TRACKING_TOKEN=${MLFLOW_TRACKING_TOKEN}"

export LOG_TO_MANTIK="True"

if [ "$NODES" -gt "1" ]; then
  python -c "import mlflow; mlflow.start_run('${SLURM_JOB_ID}')"
fi

srun apptainer run \
  -B ${REPO_DIR}/configs:/opt/vissl/configs \
  -B ${REPO_DIR}/modified/data/disk_dataset.py:/opt/vissl/vissl/data/disk_dataset.py \
  -B ${REPO_DIR}/modified/hooks/log_hooks.py:/opt/vissl/vissl/hooks/log_hooks.py \
  -B ${REPO_DIR}/modified/losses/deepclusterv2_loss.py:/opt/vissl/vissl/losses/deepclusterv2_loss.py \
  -B ${REPO_DIR}/modified/utils/distributed_launcher.py:/opt/vissl/vissl/utils/distributed_launcher.py \
  --nv \
  /p/project/deepacf/maelstrom/emmerich1/vissl.sif \
  python /opt/vissl/tools/run_distributed_engines.py \
  config=remote \
  config.DISTRIBUTED.NUM_NODES=${SLURM_NNODES} \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=${N_GPUS} \
  config.DISTRIBUTED.RUN_ID=${RUN_ID} \
  config.CHECKPOINT.DIR=${CHECKPOINTS_DIR} \
  config.DATA.TRAIN.COPY_DESTINATION_DIR=${TMP_DATA_DIR} \
  config.HOOKS.TENSORBOARD_SETUP.LOG_DIR=${TENSORBOARD_LOG_DIR} \
  config.OPTIMIZER.num_epochs=${EPOCHS} \
  config.DATA.TRAIN.DATASET_NAMES=${DATASET} \
  config.LOSS.deepclusterv2_loss.output_dir=${CLUSTERING_DIR} \
  config.OPTIMIZER.param_schedulers.lr.lengths=${LENGTHS}

status=$?

if [ "$NODES" -gt "1" ]; then
  if [ $status -eq 0 ]; then
    python -c "import mlflow; mlflow.end_run()"
  else
    python -c "import mlflow; mlflow.end_run('FAILED')"
  fi;
fi

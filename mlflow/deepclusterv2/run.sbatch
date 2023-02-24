#!/bin/bash -x
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=single-node.%j
#SBATCH --error=single-node.%j

BASE_DIR=/p/scratch/deepacf/maelstrom/emmerich1/deepclusterv2/${SLURM_JOB_ID}
CHECKPOINTS_DIR=${BASE_DIR}/checkpoints
TENSORBOARD_LOG_DIR=${BASE_DIR}/tensorboard-logs
TMP_DATA_DIR=${BASE_DIR}/tmp

N_GPUS = $((${CUDA_VISIBLE_DEVICES: -1} + 1))

mkdir -p ${BASE_DIR} ${CHECKPOINTS_DIR} ${TENSORBOARD_LOG_DIR} ${TMP_DATA_DIR}

srun apptainer run \
  -B /p/home/jusers/emmerich1/juwels/code/a6/mlflow/deepclusterv2/configs:/opt/vissl/configs \
  --nv \
  /p/project/deepacf/maelstrom/emmerich1/vissl.sif \
  python /opt/vissl/tools/run_distributed_engines.py \
  config=remote \
  config.DISTRIBUTED.NUM_NODES=${SLURM_NNODES} \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=${N_GPUS} \
  config.DISTRIBUTED.RUN_ID=${SLURM_JOB_ID} \
  config.CHECKPOINT.DIR=${CHECKPOINTS_DIR} \
  config.DATA.TRAIN.COPY_DESTINATION_DIR=${TMP_DATA_DIR} \
  config.HOOKS.TENSORBOARD_SETUP.LOG_DIR=${TENSORBOARD_LOG_DIR} \
  config.OPTIMIZER.num_epochs=20 \
# config.OPTIMIZER.lengths=[0.1, 0.9]               # 100ep FOR EPOCHS VARY .>>
# config.OPTIMIZER.lengths=[0.05, 0.95]             # 200ep
# config.OPTIMIZER.lengths=[0.025, 0.975]           # 400ep
# config.OPTIMIZER.lengths=[0.02, 0.98]             # 500ep
# config.OPTIMIZER.lengths=[0.0166667, 0.9833333]   # 600ep
# config.OPTIMIZER.lengths=[0.0125, 0.9875]         # 800ep
# config.OPTIMIZER.lengths=[0.01, 0.99]             # 1000ep
# config.OPTIMIZER.lengths=[0.0128, 0.9872]         # 1ep IG-1B
# config.OPTIMIZER.lengths=[0.00641, 0.99359]       # 2ep IG-1B
  config.OPTIMIZER.lengths=[0.002563, 0.997437]     # 5ep IG-1B = 50 ep IG-100M

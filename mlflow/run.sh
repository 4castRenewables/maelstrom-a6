srun apptainer run \
    -B ${PWD}/src/a6:/opt/a6/src/a6 \
    --nv   \
    /p/project/deepacf/maelstrom/emmerich1/a6-cuda.sif   \
    python $PWD/mlflow/train_dcv2.py           \
    --data-path /p/project/deepacf/maelstrom/emmerich1/data/deepclusterv2/daily           \
    --nmb-crops 2           \
    --size-crops 96           \
    --min-scale-crops 0.05           \
    --max-scale-crops 1.           \
    --crops-for-assign 0 1           \
    --temperature 0.1           \
    --feat-dim 128           \
    --nmb-prototypes 30 30 30 \
    --epochs 1           \
    --batch-size 64       \
    --base-lr 0.3          \
    --final-lr 0.05         \
    --dist-url "tcp://${SLURMD_NODENAME}i:29500"        \
    --arch resnet50           \
    --dump-path /p/project/deepacf/maelstrom/emmerich1/dcv2/dump
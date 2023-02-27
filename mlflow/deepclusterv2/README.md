# DeepClusterV2

This MLproject uses the DeepClusterV2 (DCv2) algorithm from the
[FaceBook VISSL library](https://github.com/facebookresearch/vissl).

## General notes

* The folder structure for the configuration has to be `configs/config`.
* Each folder and subfolder within `configs/` has to contain an empty `__init__.py`.
* The model configuration is given in `configs/config/deepclusterv2.yaml`.
* The dataset paths are configured in `configs/config/dataset_catalog.json`.

## Build and deploy the Apptainer image

Run

```bash
make deploy-vissl
```

or, alternatively

```bash
make build-vissl
make upload-vissl
```

## Testing locally

**Note**: To make the `dataset_catalog.json` available in the image, the `configs` directory
has to be bound into the container, i.e. `-B $PWD/configs:/opt/vissl/configs`.

```bash
apptainer run \
    -B $PWD/mlflow/deepclusterv2/configs:/opt/vissl/configs \
    -B $PWD/data/deepclusterv2:/data \
    --nv \
    mlflow/deepclusterv2/vissl.sif \
    python /opt/vissl/tools/run_distributed_engines.py \
    config=local
```

## Running remotely

`cd` into the git repository of A6. Then

```bash
apptainer run \
    -B $PWD/mlflow/deepclusterv2/configs:/opt/vissl/configs \
    --nv \
    /p/project/deepacf/maelstrom/emmerich1/vissl.sif \
    python /opt/vissl/tools/run_distributed_engines.py \
    config=remote
```

or submit via `sbatch`

```bash
NODES=<number_of_nodes> N_GPUS=<number_of_gpus_per_node> EPOCHS=<number_of_epochs> sbatch -A <account> --partition <partition> --nodes=${NODES} --gres=gpu:${N_GPUS} mlflow/deepclusterv2/run.sbatch
```

## Running with mantik

```bash
mantik runs submit mlflow/deepclusterv2
```

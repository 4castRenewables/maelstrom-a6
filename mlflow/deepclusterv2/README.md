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
(or `deploy-vissl-e4`)

or, alternatively

```bash
make build-vissl
make upload-vissl
```
(or `build/upload-vissl-e4`)

## Testing locally

**Note**: To make the `dataset_catalog.json` available in the image, the `configs` directory
on the local machine needs to be bound to the respective path in the image, i.e. `-B $PWD/configs:/opt/vissl/configs`.

```bash
apptainer run \
    -B $PWD/mlflow/deepclusterv2/configs:/opt/vissl/configs \
    -B $PWD/src/tests/data/deepclusterv2:/data \
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
export NODES=<number_of_nodes> N_GPUS=<number_of_gpus_per_node> EPOCHS=<number_of_epochs>
sbatch -A <account> --partition <partition> --nodes=${NODES} --gres=gpu:${N_GPUS} mlflow/deepclusterv2/run.sbatch
```

## Running with mantik

```bash
mantik runs submit mlflow/deepclusterv2
```

## Using JUBE for benchmarking

[JUBE](https://apps.fz-juelich.de/jsc/jube/jube2/docu/) can be used for benchmarking.
The benchmarks are defined in `jube.yaml`.
To run the benchmarks, use

```bash
jube run jube.yaml --tag jwc test
```

Replace the tags with the respective tags.
Available tags:

* test (single node/gpu, devel queues, small data sample)
* jwc (JUWELS Cluster)
* jwb (JUWELS Booster)
* e4 (E4 systems)
  * intel (Intel CPU + NVIDIA A100 GPU nodes)
  * amd (AMD CPU + AMD MI100 GPU nodes)
  * arm (ARM CPU + NVIDIA A100 GPU nodes)
    * v100 (ARM CPU + NVIDIA V100 GPU nodes)

*Note:*
For debugging consider the `--debug`, `--devel`, and/or `-v` options.

Once all runs are finnished, analysis can be performed via

```bash
jube analyse ap6-run/
```

## Using AMD GPUS (ROCm)

```bash
make deploy-vissl-rocm-e4
```

or, alternatively

```bash
make build-vissl-rocm
make upload-vissl-rocm-e4
```

Running with ROCm required passing
`config.OPTIMIZER.use_larc=False config.MODEL.AMP_PARAMS.USE_AMP=False`.

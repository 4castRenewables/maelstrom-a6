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
    -B $PWD/configs:/opt/vissl/configs mlflow/deepclusterv2/vissl.sif \
    python /opt/vissl/tools/run_distributed_enginges.py \
    config=local
```


## Running with mantik

```bash
mantik runs submit mlflow/deepclusterv2
```

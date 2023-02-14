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

## Training

```bash
mantik runs submit mlflow/deepclusterv2
```

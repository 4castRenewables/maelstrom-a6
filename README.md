# Maelstrom Application 6

Lifetime determination of large-scale weather regimes.

## Prerequisites

Installation of the following packages is required:

* gcc (build-essential) (required by HDBSCAN)
* libgeos and libgeos-dev (required by Cartopy)
* python3-opencv (required by opencv)

Ubuntu 20:

```bash
sudo apt-get install -y \
  build-essential \
  libgeos-3.9.0 \
  libgeos-dev \
  python3-opencv
```

Ubuntu 22:

```bash
sudo apt-get install -y \
  build-essential \
  libgeos3.10.2 \
  libgeos-dev \
  python3-dev \
  python3-opencv
```

### Using `torch-cpu` for local development

For local development, `torch-cpu` can be installed:

```shell
poetry run pip install -r requirements-cpu.txt
```

### Version conflicts

The versions of pytorch and torchvision _must_ match in all of these files:

- `pyproject.toml`
- `requirements-cpu.txt`
- `docker/a6-cuda.Dockerfile`

Otherwise, different versions might get installed, which will lead to conflicts.

## Running with MLflow

### Running directly via Python

1. Copy the `.env.example` to a `.env` file, set the required environment variables
   for tracking (see first block in `.env.example`) and then source the file:
   ```commandline
   source .env
   ```
   - **Imporant note:** Make sure to copy the `.env.example` file to a file with a `.env`
   extension. Such files will be ignored by git (see `.gitignore`). Otherwise, you
   will risk to commit your credentials to the git repository.
   - *Note:* Make sure to set the correct `MLFLOW_EXPERIMENT_ID` environment variable to
   track to the desired experiment.
2. Initialize tracking with mantik
   ```commandline
   eval $(poetry run mantik init)
   ```
   The above command will set the `MLFLOW_TRACKING_TOKEN` environment variable which enables
   tracking to mantik.
3. Run the DCv2 script
   ```commandline
   poetry run python mlflow/train_dcv2.py --enable-logging --use-cpu --epochs 1 --nmb-clusters 2
   ```
   **Note:** Running with the data used by the script as default file requires git-lfs.
   When executing for the first time, the data file has to be pulled via `git-lfs pull`.
4. Refresh the MLflow UI to see the logged parameters, metrics models and artifacts.

### Running as a project

1. Build the Docker image
   ```commandline
   make build-docker
   ```
2. Initialize tracking to mantik and set the `MLFLOW_EXPERIMENT_ID` environment variable
   (see above).
3. Run the project
   ```commandline
   poetry run mlflow run mlflow/a6 \
     -e cluster \
     -P weather_data=/data/temperature_level_128_daily_averages_2020.nc \
     -P config=cluster.yaml
     -P use_varimax=false
   ```

**Note:** The a6 package is installed into the Docker container
at build time. If the source code of the a6 package was modified,
the Docker image has to be rebuilt (see 1.) in order to have the updated source code
in the container image. The given folder (`mlflow/`), on the other hand, is copied by mlflow into
the container when running the project and, hence, does not require rebuilding the
Docker image manually if any of these files was modified
(see
[here](https://github.com/mlflow/mlflow/blob/276f71e0dfd496701774b976103dc8cce72734f2/mlflow/projects/docker.py#L60)).

### Run remotely on HPC

1. Build the Apptainer image
   ```commandline
   make build-cuda
   ```
2. Set the required environment variables for the Compute Backend:

   - `MANTIK_UNICORE_USERNAME`
   - `MANTIK_UNICORE_PASSWORD`
   - `MANTIK_COMPUTE_BUDGET_ACCOUNT`
3. Run on HPC via mantik
   ```commandline
   poetry run mantik runs submit \
     --run-name "<run-name>" \
     --entry-point dcv2 \
     --backend-config compute-backend-config-dcv2.yaml \
     $PWD/mlflow/
   ```

**Note:**
Running with Apptainer (and not as an MLproject via `mlflow run`)
does not track the git version (git commit hash), because, when creating a new run,
MLflow attempts to import the git Python module and read the project repository to
retrieve the commit hash. This is not possible inside the Apptainer container since

1. git is not installed within the container (error is usually logged by MLflow, but can be
   silenced by setting the `GIT_PYTHON_REFRESH=quiet` environment variable inside the container).
2. the repository is not available inside the container, but only the `train_kmeans.py` file.
   Hence, installing git inside the container does not solve the issue.

As a consequence, the version (`mlflow.source.git.commit` tag) is set to `None`.

## Building and deploying the Jupyter kernel

1. Prerequisites:
   - Create a private SSH file `~/.ssh/jsc` (`~/.ssh/e4`) and upload its public counterpart
     to JuDoor (the E4 help center), or adjust the path to the `JSC_SSH_PRIVATE_KEY_FILE`
     (`E4_SSH_PRIVATE_KEY_FILE`) in the `Makefile`.
   - JSC: Set the `MANTIK_UNICORE_USERNAME` and `MANTIK_UNICORE_PASSWORD` environment
     variables to allow uploading via SSH.
   - E4: Set the `E4_USERNAME` and `E4_SERVER_IP` environment variables.
     `E4_SERVER_IP` here is the IP of the E4 machine you want to use for SSH login.
2. Build Apptainer image with package and ipykernel installed
   ```commandline
   make build-jsc-kernel
   ```
   For E4, use the `build-e4-kernel` target.
3. Upload the image and the `kernel.json` file:
   ```commandline
   make upload-jsc-kernel
   ```
   For E4, use the `upload-e4-kernel` target.
   *Note*: Alternatively, you can also execute the two above steps at once:
   ```commandline
   make deploy-jsc-kernel
   ```
   For E4, use the `deploy-e4-kernel` target.

If this worked correctly, the kernel should be available in Jupyter JSC/on the E4 system
under the name `a6`.
The Apptainer image may generally be used to run the package on e.g. JUWELS
via `apptainer exec <path to image> python <path to script>`.

## Running on Juwels (Booster)

1. Start a Jupyter lab via Jupyter JSC on the respective login node (i.e. Juwels or Juwels Booster).
2. Select the kernel (see above).
3. Run the notebook `notebooks/jsc/parallel_a6.iypnb`.

## Running on E4

1. Connect to the VPN.
2. SSH onto a certain host.
3. The kernel needs Apptainer (formerly Singularity), hence the module has to be loaded
   ```commandline
   module load go-1.17.6/singularity-3.9.5
   ```
4. Start jupyter on the host
   ```commandline
   cd <repo directory>
   poetry install -E notebooks
   poetry run jupyter notebook
   ```
5. From your local terminal, establish an SSH tunnel to the machine's Jupyter port
   ```commandline
   ssh -fN -L <local port>:localhost:8888 <user>@<IP of the host>
   ```
6. Access Jupyter from your local browser by copying the token or URL from the output of
   `poetry run jupyter notebook` command. The URL should look as follows:
   `http://localhost:8888/?token=<token>`.
7. Run the notebook `notebooks/e4/parallel_a6.ipynb`.


## Known Issues

### Additional `expvar` dimension in ERA5 data

Recent ERA5 data may contain an additional dimension called `expvar` with levels `1` and `5`.
Level 1 is typically `NaN` after some point in the `time` dimension, and Level 5 is `NaN` up to that point.
After that point in time, this is the opposite: level 1 is `NaN` and level 5 has values.
Thus, the levels have to be reduced by taking the sum, ignoring `NaN`. This can be achieved with `np.nansum`:

```python
ds_new = ds.reduce(np.nansum, dim="expver", keep_attrs=True)
```

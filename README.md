# lifetime-determination

Lifetime determination of large-scale weather regimes.

## Building and deploying the Jupyter kernel

1. Build Singularity image with package and ipykernel installed
   ```commandline
   sudo singularity build notebooks/<e4 or juwels>/jupyter_kernel.sif notebooks/<e4 or juwels>/jupyter_kernel_recipe.def
   ```
2. Prepare uploading the Singularity image:
     - JUWELS: create a folder in `/p/scratch/deepacf/<JUDOOR user name>/jupyter-lifetimes`.
     - E4: Create a folder in your home directory, e.g. `/home/<user>/.singularity`
       On the E4 systems, Singularity images cannot be run from shared partitions such as
       the `/data/maelstrom` partition.
3. The kernel specification in `kernel.json` must point to the image.
   Hence, the path in the `argv` section must be adapted to point to the image:
     - JUWELS: `/p/scratch/deepacf/<JUDOOR user name>/jupyter-lifetimes/jupyter-kernel.sif`.
     - E4: e.g. `/home/<user>/.singularity/`
   This file must be uploaded to
   `$HOME/.local/share/jupyter/kernels/<kernel name>/kernel.json`
   where `<kernel display name>` is the name under which the kernel will appear in the Jupyter UI.
   This will be done in the following step.
4. Upload to JUWELS/E4 (e.g. somewhere to scratch) using `notebooks/<e4 or juwels>/upload_kernel.sh`,
   but make sure to update the `SSH_PRIVATE_KEY_FILE` and `JUDOOR_USER`
   variables in the script prior to execution. Then upload by executing
   ```commandline
   bash notebooks/<e4 or juwels>/upload_kernel.sh
   ```

If this worked correctly, the kernel should be available in Jupyter JSC/on the E4 system.
The Singularity image may generally be used to run the package on e.g. JUWELS
via `singularity exec <path to image> python <path to script>`.

# Running on Juwels (Booster)

1. Start a Jupyter lab via Jupyter JSC on the respective login node (i.e. Juwels or Juwels Booster).
2. Select the kernel (see above).
3. Run the notebook `notebooks/juwels/parallel_lifetimes.iypnb`.

# Running on E4

1. Connect to the VPN.
2. SSH onto a certain host.
3. The kernel needs Singularity, hence the module has to be loaded
   ```commandline
   module load go-1.17.6/singularity-3.9.5
   ```
4. Start jupyter on the host
   ```commandline
   cd code/lifetime-determination
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
7. Run the notebook `notebooks/e4/parallel_lifetimes.ipynb`.

# Running with MLflow

## Running directly via Python
1. Install via `poetry install`.
2. Start the MLflow UI in a separate terminal via `poetry run mlflow ui`.
3. Run the MLflow script via
   ```commandline
   poetry run python mlflow/main.py \
     --data data/temperature_level_128_daily_averages_2020.nc \
     --variance-ratios 0.8 0.9 \
     --n-clusters 3 4 \
     --use-varimax False
   ```
4. Refresh the MLflow UI to see the logged parameters, metrics models and artifacts.

### Running as a project

1. Build the Docker image
   ```commandline
   docker build -f mlflow/Dockerfile -t lifetimes-mlflow:latest .
   ```
2. Run the project
   ```commandline
   poetry run mlflow run mlflow \
     -P data=data/temperature_level_128_daily_averages_2020.nc \
     -P variance_ratio=0.95 \
     -P n_clusters=4 \
     -P use_varimax=True
   ```
   In order to run with a set of parameters execute as follows:

   ```commandline
   poetry run mlflow run mlflow \
     -P data=data/temperature_level_128_daily_averages_2020.nc \
     -P variance_ratio="0.9 0.95" \
     -P n_clusters="3 4" \
     -P use_varimax="False True"
   ```
   **Note:** When using multiple values for the parameters, the cartesian
   product is build from these to run every possible combination of input values.
   This is done in the `mlflow/main.py`, though, and is not a feature of mlflow.

**Note:** The lifetimes package is installed into the Docker container
at build time. If the source code of the lifetimes package was modified,
the Docker image has to be rebuilt (1.) in order to have the updated source code
in the container image. The `main.py`, on the other hand, is copied by mlflow into
the container when running the project and, hence, does not require rebuilding the
Docker image manually if the file was modified.

In fact, mlflow copies the whole current working directory into a new container
image based on the image build in 1.
(see
[here](https://github.com/mlflow/mlflow/blob/276f71e0dfd496701774b976103dc8cce72734f2/mlflow/projects/docker.py#L60)),
which includes the `data` directory as well. This is the reason the data are
available in the container at runtime.

### Run manually on HPC

1. Build the Docker image (see step 1 above).
2. Build the Singularity image
   ```commandline
   sudo singularity build lifetimes-mlflow.sif mlflow/recipe.def
   ```
3. Test locally
   ```commandline
   singularity run \
     --cleanenv \
     --env MLFLOW_TRACKING_URI=file://${PWD}/mlruns \
     lifetimes-mlflow.sif \
     --data /opt/data/temperature_level_128_daily_averages_2020.nc \
     --variance-ratios 0.95 \
     --n-clusters 4 \
     --use-varimax True
   ```
4. Copy image to JUWELS
   ```commandline
   scp lifetimes-mlflow.sif <JUDOOR user>@juwels.fz-juelich.de:~
   ```
5. SSH onto JUWELS and run using the same command as in step 3 but via Slurm
   ```commandline
   srun -A <project> -p <batch/devel> <command from 3.>
   ```
   The mlflow logs are then written to an `mlruns` folder located
   at the current path (`$PWD`).

**Note:** The above procedure (i.e. the building of the Singularity image)
copies the source file (`main.py`) during build time into the container image.
Thus, if `main.py` was modified, the image has to be rebuilt to have the changes
in the image. This, of course, applies to the lifetimes package as well.

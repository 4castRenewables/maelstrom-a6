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

1. Install via `poetry install`.
2. Start the MLflow UI in a separate terminal via `poetry run mlflow ui`.
3. Run the MLflow script via `poetry run python scripts/mlflow/main.py`.
4. Refresh the MLflow UI to see the logged parameters, metrics models and artifacts.

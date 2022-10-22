# lifetime-determination

Lifetime determination of large-scale weather regimes.

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
   eval $(mantik init)
   ```
   The above command will set the `MLFLOW_TRACKING_TOKEN` environment variable which enables
   tracking to mantik.
3. Run a command (e.g. `train kmeans`) via
   ```commandline
   poetry run a6 train kmeans \
     --weather-data data/temperature_level_128_daily_averages_2020.nc \
     --n-components-start 3 \
     --n-components-end 4 \
     --n-clusters-start 3 \
     --n-clusters-end 4
   ```
   Pass `--use-varimax` to use VariMax rotation for the PCs.
   **Note:** Running with the above data file requires git-lfs.
   When executing for the first time, the data file has to be pulled via `git-lfs pull`.
4. Refresh the mantik UI to see the logged parameters, metrics models and artifacts.

### Running as a project

1. Build the Docker image
   ```commandline
   make build-docker
   ```
2. Initialize tracking to mantik and set the `MLFLOW_EXPERIMENT_ID` environment variable
   (see above).
3. Run the project
   ```commandline
   poetry run mlflow run mlflow \
     -e kmeans \
     -P data=/data/temperature_level_128_daily_averages_2020.nc \
     -P n_components=3 \
     -P n_clusters=4 \
     -P use_varimax=False
   ```
   In order to run with a set of parameters execute as follows:
   ```commandline
   poetry run mlflow run mlflow \
     -e kmeans \
     -P data=/data/temperature_level_128_daily_averages_2020.nc \
     -P n_components_start=3 \
     -P n_components_end=4 \
     -P n_clusters_start=3 \
     -P n_clusters_end=4 \
     -P use_varimax=--use-varimax
   ```
   **Note:** When using multiple values for the parameters, the cartesian
   product is build from these to run every possible combination of input values.
   This is done in the `mlflow/train_kmeans.py`, though, and is not a feature of mlflow.

**Note:** The a6 package is installed into the Docker container
at build time. If the source code of the a6 package was modified,
the Docker image has to be rebuilt (see 1.) in order to have the updated source code
in the container image. The `train_kmeans.py`, on the other hand, is copied by mlflow into
the container when running the project and, hence, does not require rebuilding the
Docker image manually if the file was modified.

In fact, mlflow copies the whole project folder into a new container
image based on the image build in 1.
(see
[here](https://github.com/mlflow/mlflow/blob/276f71e0dfd496701774b976103dc8cce72734f2/mlflow/projects/docker.py#L60)).

### Run remotely on HPC

1. Build the Docker image (see step 1 above).
2. Build the Singularity image
   ```commandline
   make build
   ```
3. Test locally with remote tracking:
   ```commandline
   singularity run \
     mlflow/a6-mlflow.sif \
     python /opt/train_kmeans.py \
     --data ${PWD}/data/temperature_level_128_daily_averages_2020.nc \
     --n-components 3 \
     --n-clusters 4 \
     --use-varimax True
   ```
4. Set the required environment variables for the Compute Backend:
   ```bash
   export MANTIK_UNICORE_USERNAME=<JuDoor username>
   export MANTIK_UNICORE_PASSWORD=<JuDoor password>
   export MANTIK_UNICORE_PROJECT=<JuDoor project>
   ```
5. Run on HPC via mantik
   ```commandline
   poetry run mantik mlflow \
     --experiment-id <experiment ID> \
     --entry-point kmeans \
     -P data="/opt/data/temperature_level_128_daily_averages_2020.nc" \
     -P n_components=3 \
     -P n_clusters=4 \
     -P use_varimax=False
   ```

**Notes:**
- The above procedure (i.e. the building of the Singularity image)
  installs the package during build time into the container image.
  Thus, the package was modified, the image has to be rebuilt to have the changes
  in the image.
- Running with Singularity (and not as an MLproject via `mlflow run`)
  does not track the git version (git commit hash), because, when creating a new run,
  MLflow attempts to import the git Python module and read the project repository to
  retrieve the commit hash. This is not possible inside the Singularity container since
  1. git is not installed within the container (error is usually logged by MLflow, but can be
     silenced by setting the `GIT_PYTHON_REFRESH=quiet` environment variable inside the container).
  2. the repository is not available inside the container, but only the `train_kmeans.py` file.
     Hence, installing git inside the container does not solve the issue.

  As a consequence, the version (`mlflow.source.git.commit` tag) is set to `None`.

### Deployment and Inference with Amazon SageMaker

1. Register a model in the MLflow UI.
2. Build and push the MLflow image for serving with Amazon Sagemaker
   ```commandline
   poetry run mlflow sagemaker build-and-push-container
   ```
3. Run the MLflow
   [`deployments create`](https://mlflow.org/docs/latest/python_api/mlflow.sagemaker.html#mlflow-sagemaker)
   entrypoint
   ```commandline
   poetry run python mlflow/deploy.py \
       --endpoint-name a6 \
       --image-uri <URI to ECR image> }
       --model-uri "models:/<registered model name>/<version>" \
       --role <SageMaker role ARN> \
       --bucket <S3 bucket Artifact Storage name> \
       --vpc-config '{"SecurityGroupIds": ["<MLflow VPC security group ID>"], "Subnets": ["<MLflow VPC private subnet ID>"]}'
   ```
   The SageMaker role has to be created
   (under `User Menu > Security credentials > Roles > Create Role > AWS account`)
   and needs the following permissions:
   - AmazonS3ReadOnlyAccess
   - AmazonSageMakerFullAccess
4. Run the inference per the deployed SageMaker endpoint
   ```commandline
   poetry run python mlflow/infer.py \
       --endpoint-name a6 \
       --data $PWD/data/temperature_level_128_daily_averages_2020.nc
       --variance-ratio 0.95
       --use-varimax False
   ```
   *Note:* Take care how many input features the model requires.
   It may be required to use the exact same `variance_ratio` as was used
   for training the respective model.

## Building and deploying the Jupyter kernel

1. Prerequisites:
   - Create a private SSH file `~/.ssh/jsc` (`~/.ssh/e4`) and upload its public counterpart
     to JuDoor (the E4 help center), or adjust the path to the `JSC_SSH_PRIVATE_KEY_FILE`
     (`E4_SSH_PRIVATE_KEY_FILE`) in the `Makefile`.
   - JSC: Set the `MANTIK_UNICORE_USERNAME` and `MANTIK_UNICORE_PASSWORD` environment
     variables to allow uploading via SSH.
   - E4: Set the `E4_USERNAME` and `E4_SERVER_IP` environment variables.
     `E4_SERVER_IP` here is the IP of the E4 machine you want to use for SSH login.
2. Build Singularity image with package and ipykernel installed
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
The Singularity image may generally be used to run the package on e.g. JUWELS
via `singularity exec <path to image> python <path to script>`.

## Running on Juwels (Booster)

1. Start a Jupyter lab via Jupyter JSC on the respective login node (i.e. Juwels or Juwels Booster).
2. Select the kernel (see above).
3. Run the notebook `notebooks/jsc/parallel_a6.iypnb`.

## Running on E4

1. Connect to the VPN.
2. SSH onto a certain host.
3. The kernel needs Singularity, hence the module has to be loaded
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

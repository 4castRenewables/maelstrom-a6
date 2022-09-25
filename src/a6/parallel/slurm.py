import contextlib
import logging
import os
import pathlib
import typing as t

import a6.parallel._client as _client
import a6.parallel.types as types
import dask.distributed
import dask_jobqueue

logger = logging.getLogger(__name__)

SINGULARITY_IMAGE_ENV_VAR = "SINGULARITY_IMAGE"


class DaskSlurmClient(_client.Client):
    """Allows parallel execution of jobs."""

    _interface: t.Optional[str] = None
    _shebang = "#!/usr/bin/env bash"
    _local_directory: t.Optional[str] = None
    _death_timeout = "30s"
    _default_log_directory_name = "job_logs"
    _python_executable = "srun singularity run {} python"
    _client: t.Optional[dask.distributed.Client] = None
    _extra_job_commands: t.Optional[list[str]] = None
    _extra_worker_commands: t.Optional[list[str]] = None

    def __init__(
        self,
        queue: str,
        project: str,
        cores: int = 96,
        processes: int = 1,
        memory: int = 90,
        walltime: str = "00:30:00",
        log_directory: t.Optional[str] = None,
        scheduler_options: t.Optional[dict] = None,
        dashboard_port: int = 56755,
        python_executable: t.Optional[str] = None,
        extra_job_commands: t.Optional[list[str]] = None,
        network_interface: t.Optional[str] = None,
        extra_slurm_options: t.Optional[list[str]] = None,
        **kwargs,
    ):
        """Initialize the cluster and client.

        Parameters
        ----------
        queue : str
            Queue to deploy the jobs.
        project : str
            Project to bill.
        cores : int, default=48
            Number of cores per job.
        processes : int, default=1
            Number of processes the job is split up into.
        memory : int, default=90,
            Memory in GB per job.
        walltime : str, default="00:30:00"
            Walltime for each worker.
        log_directory : str, optional
            Directory to write the logs of each worker.
            Defaults to `"$HOME/job_logs/"`.
        scheduler_options : dict, optional
            Additional arguments to pass to the dask worker.
        dashboard_port : int, default=56755
            Port to use for the dask dashboard.
        python_executable : str,
            default="singularity run $SINGULARITY_IMAGE python"
            Python executable to use in each job.
        extra_job_commands : list[str], optional
            Extra bash commands to execute in the batch script before launching
            each job.
        network_interface : str, optional
            Network interface to use.
        extra_slurm_options : list[str], optional
            Extra Slurm options to include in the batch script.
            Each option will be added to the beginning of the script with the
            `#SBATCH` prefix.
        kwargs
            Additional parameters passed to `dask_jobqueue.SLURMCluster`.

        """
        if log_directory is None:
            log_directory = (
                self._create_default_log_directory_path_in_home_directory()
            )
        _create_directory_with_parents(log_directory)

        if python_executable is None:
            python_executable = self._create_python_executable_from_env_var()

        if scheduler_options is None:
            scheduler_options = {}
        scheduler_options |= {"dashboard_address": f":{dashboard_port}"}

        extra_job_commands = self._update_extra_job_commands(extra_job_commands)

        logger.debug("Creating cluster: %s", locals())

        self._queue = queue
        self._project = project
        self._cores = cores
        self._processes = processes
        self._memory = _memory_in_gb(memory)
        self._scheduler_options = scheduler_options
        self._walltime = walltime
        self._log_directory = log_directory
        self._interface = network_interface or self._interface
        self._python = python_executable
        self._env_extra = extra_job_commands
        self._job_extra = extra_slurm_options
        self._kwargs = kwargs

        self._cluster = self._create_cluster()

    def __enter__(self) -> "DaskSlurmClient":
        """Wait for the engines."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cancel jobs and shut down the client."""
        self._close_cluster_and_client()

    @property
    def ready(self) -> bool:
        """Return whether the engines are ready."""
        return self._client is not None

    @property
    def job_script(self) -> str:
        """Return the job script."""
        return self._cluster.job_script()

    @property
    def log_directory(self) -> str:
        """Return the log directory path."""
        return self._log_directory

    @property
    def cluster(self) -> dask_jobqueue.SLURMCluster:
        """Return the Slurm cluster instance."""
        return self._cluster

    @contextlib.contextmanager
    def scale(
        self,
        workers: int = 1,
        jobs: t.Optional[int] = None,
        cores: t.Optional[int] = None,
        memory: t.Optional[int] = None,
    ):
        """Scale up the workers.

        Parameters
        ----------
        workers : int, default=1
            Number of workers to start.
        jobs : int, default=1
            Number of processes the job is split up into.
        cores : int, optional
            Number of cores per job.
        memory : int, optional
            Memory in GB per job.

        Yields
        ------
        DaskSlurmClient

        """
        if memory is not None:
            memory = _memory_in_gb(memory)

        logger.debug("Establishing client: %s", locals())

        self._client = dask.distributed.Client(self._cluster)
        self._cluster.scale(n=workers, jobs=jobs, cores=cores, memory=memory)
        yield self
        self._close_cluster_and_client()

    def _close_cluster_and_client(self) -> None:
        logger.debug("Closing cluster and connection")

        # Close cluster and workers.
        self._cluster.close()
        self._client.close()

        # Reset cluster.
        self._client = None
        self._cluster = self._create_cluster()

    def _create_default_log_directory_path_in_home_directory(self) -> str:
        home_path = os.environ.get("HOME")
        log_directory = f"{home_path}/{self._default_log_directory_name}"
        return log_directory

    def _create_python_executable_from_env_var(self) -> str:
        singularity_image = os.environ.get(SINGULARITY_IMAGE_ENV_VAR)
        if singularity_image is None:
            raise RuntimeError(
                f"Environment variable '{SINGULARITY_IMAGE_ENV_VAR}' is not set"
            )
        python_executable = self._python_executable.format(singularity_image)
        return python_executable

    def _update_extra_job_commands(
        self, extra_job_commands: t.Optional[list[str]]
    ) -> t.Optional[list[str]]:
        if extra_job_commands is None:
            extra_job_commands = []

        if self._extra_job_commands is not None:
            return [*extra_job_commands, *self._extra_job_commands]
        return extra_job_commands

    def _create_cluster(self) -> dask_jobqueue.SLURMCluster:
        local_directory = self._create_local_directory_path()
        return dask_jobqueue.SLURMCluster(
            queue=self._queue,
            project=self._project,
            cores=self._cores,
            processes=self._processes,
            memory=self._memory,
            shebang=self._shebang,
            scheduler_options=self._scheduler_options,
            walltime=self._walltime,
            local_directory=local_directory,
            death_timeout=self._death_timeout,
            log_directory=self._log_directory,
            interface=self._interface,
            python=self._python,
            extra=self._extra_worker_commands,
            env_extra=self._env_extra,
            job_extra=self._job_extra,
            **self._kwargs,
        )

    def _create_local_directory_path(self) -> str:
        if self._local_directory is None:
            home_directory = os.environ.get("HOME")
            return f"{home_directory}/tmp"
        return self._local_directory

    def _execute(
        self, method: types.Method, arguments: types.Arguments
    ) -> list[t.Any]:
        if not self.ready:
            raise RuntimeError("No worker(s) running")

        delayed = (dask.delayed(method)(*args) for args in arguments)
        futures = dask.persist(*delayed)
        results = dask.compute(*futures)
        return results


def _create_directory_with_parents(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class JuwelsClient(DaskSlurmClient):
    """Dedicated client for the JUWELS cluster."""

    # TODD: Use `srun` for exeuting commands
    # Batch scripts typically use `srun` to execute commands. On JUWELS,
    # though, this leads to the error
    # `srun: error: Security violation, slurm message from uid 1028`,
    # which could not be fixed yet. This results from the batch script
    # being run from within the Singularity container. Outside the container,
    # i.e. via Terminal, the command works with `srun`. This error must thus
    # result from the Singularity container and should likely be fixable.
    _python_executable = "singularity run {} python"
    _interface = "ib0"
    _local_directory = "/tmp"


class E4Client(DaskSlurmClient):
    """Dedicated client for the E4 cluster."""

    _extra_job_commands = [
        "module load go-1.17.6/singularity-3.9.5",
        # Loading `gcc-8.5.0/slurm-21.08.4` does not allow using certain
        # partitions on the E4 system.
        "module load slurm",
        # Running the Singularity container caused the following error:
        # `FATAL:   container creation failed: failed to resolve session
        # directory /tmp/singularity-3.9.5/state/singularity/mnt/session:
        # lstat /tmp/singularity-3.9.5: no such file or directory`, which
        # is due to the given directory not existing. Thus, it has to
        # be created.
        "mkdir -p  /tmp/singularity-3.9.5/state/singularity/mnt/session",
    ]
    # For the E4 cluster, specifying the network interface does not work.
    # Using `SLURMD_NODENAME` as host for the dask workers works, though.
    _extra_worker_commands = ["--host ${SLURMD_NODENAME}"]
    # On E4, the `/data` directory is not being mounted into the container.
    _python_executable = "srun singularity run -B /data:/data {} python"


def _memory_in_gb(memory: int) -> str:
    return f"{memory}GB"

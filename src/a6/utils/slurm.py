import logging
import os

logger = logging.getLogger(__name__)


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_number_of_nodes() -> int:
    return int(os.getenv("SLURM_NNODES", 1))


def get_node_id() -> str:
    return os.getenv("SLURM_NODEID", "0")


def get_slurm_job_id() -> str:
    return os.getenv("SLURM_JOB_ID", "unknown")


def get_slurm_env_vars() -> dict[str, str | None]:
    return {
        "SLURM_JOB_ID": get_slurm_job_id(),
        "SLURM_JOB_NAME": os.getenv("SLURM_JOB_NAME"),
        "SLURM_JOB_ACCOUNT": os.getenv("SLURM_JOB_ACCOUNT"),
        "SLURM_CLUSTER_NAME": os.getenv("SLURM_CLUSTER_NAME"),
        "SLURM_JOB_PARTITION": os.getenv("SLURM_JOB_PARTITION"),
        "SLURM_JOB_NUM_NODES": os.getenv("SLURM_JOB_NUM_NODES"),
        "SLURM_NODELIST": os.getenv("SLURM_NODELIST"),
        "SLURM_JOB_CPUS_PER_NODE": os.getenv("SLURM_JOB_CPUS_PER_NODE"),
        "SLURM_GPUS_ON_NODE": os.getenv("SLURM_GPUS_ON_NODE"),
        "SLURM_CPUS_PER_TASK": os.getenv("SLURM_CPUS_PER_TASK"),
        "SLURM_NNODES": os.getenv("SLURM_NNODES"),
        "SLURM_NPROCS": os.getenv("SLURM_NPROCS"),
        "SLURM_NTASKS": os.getenv("SLURM_NTASKS"),
        "SLURM_JOB_GPUS": os.getenv("SLURM_JOB_GPUS"),
        "SLURM_JOB_STDOUT": os.getenv("SLURM_JOB_STDOUT"),
        "SLURM_JOB_STDERR": os.getenv("SLURM_JOB_STDERR"),
    }


def get_stdout_file() -> str | None:
    return os.getenv("SLURM_JOB_STDOUT")


def get_stderr_file() -> str | None:
    return os.getenv("SLURM_JOB_STDERR")

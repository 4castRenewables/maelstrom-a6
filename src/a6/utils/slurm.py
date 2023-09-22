import logging
import os

import torch

logger = logging.getLogger(__name__)


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_daemon_node_name() -> str:
    return os.getenv("SLURMD_NODENAME")


def get_number_of_nodes() -> int:
    return int(os.getenv("SLURM_NNODES", 1))


def get_node_id():
    return int(os.getenv("SLURM_NODEID", 0))


def get_global_rank(args) -> int:
    return get_node_id() * get_gpus_per_node() + args.local_rank


def get_world_size() -> int:
    n_nodes = get_number_of_nodes()
    gpus_per_node = get_gpus_per_node()
    size = n_nodes * gpus_per_node if gpus_per_node > 0 else n_nodes
    logger.info("Detected world size of %s devices", size)
    return size


def get_gpus_per_node() -> int:
    if torch.cuda.is_available():
        gpus_per_node = torch.cuda.device_count()
    elif visible_devices := os.getenv("CUDA_VISIBLE_DEVICES", ""):
        gpus_per_node = len(visible_devices.split(","))
    else:
        gpus_per_node = 0
    logger.info("Found %s GPUs available per node", gpus_per_node)
    return gpus_per_node


def get_slurm_job_id() -> str:
    return os.getenv("SLURM_JOB_ID")


def get_slurm_env_vars() -> dict[str, str]:
    return {
        "SLURM_JOB_ID": get_slurm_job_id(),
        "SLURM_JOB_NAME": os.getenv("SLURM_JOB_NAME"),
        "SLURM_JOB_ACCOUNT": os.getenv("SLURM_JOB_ACCOUNT"),
        "SLURM_CLUSTER_NAME": os.getenv("SLURM_CLUSTER_NAME"),
        "SLURM_JOB_PARTITION": os.getenv("SLURM_JOB_PARTITION"),
        "SLURM_JOB_NUM_NODES": os.getenv("SLURM_JOB_NUM_NODES"),
        "SLURM_NODELIST": os.getenv("SLURM_NODELIST"),
        "SLURM_JOB_CPUS_PER_NODE": os.getenv("SLURM_JOB_CPUS_PER_NODE"),
        "SLURM_CPUS_PER_TASK": os.getenv("SLURM_CPUS_PER_TASK"),
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

import os


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def get_node_id(node_id: int = 0):
    return int(os.getenv("SLURM_NODEID", node_id))


def get_rank(local_rank: int) -> int:
    return get_node_id() * get_gpus_per_node() + local_rank


def get_world_size() -> int:
    n_nodes = int(os.getenv("SLURM_NNODES", 1))
    gpus_per_node = get_gpus_per_node()

    if gpus_per_node > 0:
        return n_nodes * gpus_per_node

    return n_nodes


def get_gpus_per_node() -> int:
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if visible_devices is None:
        return 0
    return len(visible_devices.split(","))


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

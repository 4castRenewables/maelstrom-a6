import os


def get_node_id(node_id: int = 0):
    return int(os.getenv("SLURM_NODEID", node_id))


def get_world_size() -> int:
    n_nodes = int(os.getenv("SLURM_NNODES", 1))
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")

    if visible_devices is not None:
        n_devices = len(visible_devices.split(","))
        return n_nodes * n_devices

    return n_nodes


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

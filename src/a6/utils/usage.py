import logging
import os
import resource

import mantik.mlflow
import psutil
import torch

logger = logging.getLogger(__name__)


def log_gpu_memory_usage(device: torch.device, epoch: int, track_to_mlflow: bool):
    available = _bytes_to_gb(
        torch.cuda.get_device_properties(device).total_memory
    )
    logger.info("GPU memory available: %.5f GB", available)

    # Reserved memory is memory managed by the caching allocator
    current_reserved = _bytes_to_gb(torch.cuda.memory_reserved(device))
    max_reserved = _bytes_to_gb(torch.cuda.max_memory_reserved(device))

    logger.info(
        "GPU memory reserved current %.5f GB, max %.5f GB",
        current_reserved,
        max_reserved,
    )

    # Allocated memory is memory actually occupied by tensors
    current_usage = _bytes_to_gb(torch.cuda.memory_allocated(device))
    max_usage = _bytes_to_gb(torch.cuda.max_memory_allocated(device))

    logger.info(
        "GPU memory usage: current %.5f GB, max %.5f GB",
        current_usage,
        max_usage,
    )

    if track_to_mlflow:
        mantik.mlflow.log_metrics(
            {
                "gpu_mem_reserved_current_gb": current_reserved,
                "gpu_mem_reserved_max_gb": max_reserved,
                "gpu_mem_usage_current_gb": current_usage,
                "gpu_mem_usage_max_gb": max_usage,
            },
            step=epoch,
        )


def log_cpu_memory_usage(epoch: int, track_to_mlflow: bool):
    """Logs the current and maximum memory useage of this process."""
    current_usage = _bytes_to_gb(get_memory_usage())
    max_usage = _bytes_to_gb(get_max_memory_usage())
    logger.info(
        "CPU memory usage: current %.5f GB, max %.5f GB",
        current_usage,
        max_usage,
    )

    if track_to_mlflow:
        mantik.mlflow.log_metrics(
            {
                "cpu_mem_usage_current_gb": current_usage,
                "cpu_mem_usage_max_gb": max_usage,
            },
            step=epoch,
        )



def _bytes_to_gb(value: float) -> float:
    return value / 1024**3


def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        mem += child.memory_info().rss
    return mem


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000

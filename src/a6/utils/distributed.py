import logging
import os
import socket

import torch.distributed

import a6.utils.slurm as slurm

logger = logging.getLogger(__name__)

# Default to GPU 0
_cuda_device_index: int = 0

# Setting _cuda_device_index to -1 internally implies that we should use CPU
_CPU_DEVICE_INDEX = -1
_PRIMARY_RANK = 0


def init_distributed_mode(args, local_rank: int, initial: bool):
    """
    Initialize the following variables:
        - world_size
        - rank
    """
    args.local_rank = local_rank

    if slurm.is_slurm_job():
        args.rank = get_rank()
        args.world_size = slurm.get_world_size()
    else:
        # multi-GPU job (local or multi-node) - jobs started with
        # torch.distributed.launch read environment variables
        if "RANK" not in os.environ:
            logger.warning("RANK unset, using default value")
        if "WORLD_SIZE" not in os.environ:
            logger.warning("WORLD_SIZE unset, using default value")
        args.rank = int(os.getenv("RANK", 0))
        args.world_size = int(os.getenv("WORLD_SIZE", 1))

    os.environ["RANK"] = str(args.rank)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    logger.info(
        (
            "Required env vars for distributed mode set: "
            "RANK=%s, LOCAL_RANK=%s, WORLD_SIZE=%s"
        ),
        args.rank,
        args.rank,
        args.local_rank,
        args.world_size,
    )

    if not initial and not torch.distributed.is_initialized():
        logger.warning(
            "Distributed not initialized, initializing process group"
        )
        if args.use_cpu:
            logger.warning("Initializing CPU backend")
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(_find_free_tcp_port())
            torch.distributed.init_process_group(backend="gloo")
        else:
            logger.warning("Initializing GPU backend")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )


def get_device(args):
    return torch.device("cpu" if args.use_cpu else f"cuda:{args.local_rank}")


def get_primary_rank() -> int:
    return _PRIMARY_RANK


def is_primary_device() -> bool:
    return get_rank() == _PRIMARY_RANK


def get_rank() -> int:
    return torch.distributed.get_rank() if _is_distributed() else 0


def get_world_size() -> int:
    return torch.distributed.get_world_size() if _is_distributed() else 1


def _is_distributed() -> bool:
    return (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )


def get_cuda_device_index() -> int:
    return _cuda_device_index


def set_cuda_device_index(idx: int) -> None:
    global _cuda_device_index
    _cuda_device_index = idx
    torch.cuda.set_device(_cuda_device_index)


def set_cpu_device() -> None:
    global _cuda_device_index
    _cuda_device_index = _CPU_DEVICE_INDEX


def _find_free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def gather_from_all_ranks(tensor: torch.Tensor) -> torch.Tensor:
    gathered_tensors = _gather_tensors_from_all(tensor)
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def _gather_tensors_from_all(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Wrapper over torch.distributed.all_gather for performing
    'gather' of 'tensor' over all processes in both distributed /
    non-distributed scenarios.
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if _is_distributed_training_run():
        tensor, orig_device = _convert_to_distributed_tensor(tensor)
        gathered_tensors = [
            torch.zeros_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(gathered_tensors, tensor)
        gathered_tensors = [
            _convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]

    return gathered_tensors


def _is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


def _convert_to_normal_tensor(
    tensor: torch.Tensor, orig_device: str
) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def _convert_to_distributed_tensor(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)

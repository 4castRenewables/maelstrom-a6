import logging
import os
import socket

import numpy as np
import torch.distributed

import a6.utils.slurm as slurm

logger = logging.getLogger(__name__)


def get_global_rank(args) -> int:
    return slurm.get_global_rank(args)


def setup(args) -> None:
    logging.info(
        "Spawning process for node %s, local rank %s, global rank: %s",
        args.node_id,
        args.local_rank,
        args.global_rank,
    )
    _fix_random_seeds(args.seed)
    _get_dist_url_and_set_master_env_vars(args)
    _init_process_group(args)
    _set_device(args)


def set_required_env_vars(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """
    args.global_rank = _get_and_set_env_var("RANK", default=0)
    args.local_rank = _get_and_set_env_var("LOCAL_RANK", default=0)
    args.world_size = _get_and_set_env_var("WORLD_SIZE", default=1)


def _get_and_set_env_var(name: str, default: int | str) -> int | str:
    value = os.getenv(name)
    if value is None:
        print(
            f"WARNING: Environment variable {name!r} unset, "
            "using default value {default}"
        )
        return default
    return type(default)(value)


def _fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _get_dist_url_and_set_master_env_vars(args) -> str | None:
    print(os.environ["MASTER_ADDR"])
    print(os.environ["MASTER_PORT"])
    default_port = 29500 if _is_multi_node() else _find_free_tcp_port()
    host = _get_and_set_env_var("MASTER_ADDR", default="127.0.0.1")
    port = _get_and_set_env_var("MASTER_PORT", default=default_port)

    if not _is_multi_node():
        host = "127.0.0.1"
    elif ".juwels" in host:
        # On JUWELS, hosts get resolved by appending an i to the hostname
        host = f"{slurm.get_daemon_node_name()}i"

    os.environ["MASTER_ADDR"] = host
    os.environ["MASTER_PORT"] = str(port)

    args.dist_url = f"tcp://{host}:{port}"

    logger.info("Distributed URL is %s", args.dist_url)


def _is_multi_node() -> bool:
    return slurm.is_slurm_job() and slurm.get_number_of_nodes() > 1


def _find_free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _init_process_group(args) -> None:
    if not torch.distributed.is_initialized():
        logger.warning(
            "Distributed not initialized, initializing process group"
        )
        if args.use_cpu:
            logger.warning("Initializing CPU backend")
            torch.distributed.init_process_group(backend="gloo")
        else:
            logger.warning(
                (
                    "Initializing GPU backend using init_method=%s, "
                    "world_size=%s, rank=%s"
                ),
                args.dist_url,
                args.world_size,
                args.global_rank,
            )
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=args.dist_url,
                rank=args.global_rank,
                world_size=args.world_size,
            )
    else:
        logger.warning(
            "Torch distributed has already been initialized, "
            "reusing existing configuration"
        )


def _set_device(args) -> None:
    if not args.use_cpu and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        # perform a dummy all-reduce to initialize the NCCL communicator
        torch.distributed.all_reduce(torch.zeros(1).cuda())


def get_device(args) -> torch.device:
    device = "cpu" if args.use_cpu else f"cuda:{args.local_rank}"
    logger.info(
        "Rank %s with local rank %s using device %s",
        args.global_rank,
        args.local_rank,
        device,
    )
    return torch.device(device)


def is_primary_device() -> bool:
    return int(os.getenv("RANK")) == 0


def destroy() -> None:
    logger.info("Destroying process group")
    torch.distributed.destroy_process_group()


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

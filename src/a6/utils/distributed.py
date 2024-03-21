import dataclasses
import logging
import os
import random
import socket
import pathlib
from typing import Callable

import numpy as np
import torch.nn as nn
import torch.distributed
import torch.multiprocessing
import torch.utils.data
import torchvision.datasets

import a6.utils.slurm as slurm
import a6.utils.logging as utils_logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Properties:
    node_id: str
    dist_url: str
    local_rank: int
    global_rank: int
    world_size: int
    use_cpu: bool = False
    use_nccl: bool = True
    enable_tracking: bool = True
    verbose_logging: bool = False
    logs_filepath: pathlib.Path | None = None

    @classmethod
    def from_env(
        cls,
        use_cpu: bool = False,
        use_nccl: bool = True,
        enable_tracking: bool = True,
        verbose_logging: bool = False,
        logs_filepath: pathlib.Path | None = None,
    ) -> "Properties":
        env_vars = get_and_set_required_env_vars()
        return cls(
            node_id=slurm.get_node_id(),
            dist_url=get_dist_url_and_set_master_env_vars(),
            local_rank=env_vars.local_rank,
            global_rank=env_vars.global_rank,
            world_size=env_vars.world_size,
            use_cpu=use_cpu,
            use_nccl=use_nccl,
            enable_tracking=enable_tracking,
            verbose_logging=verbose_logging,
            logs_filepath=logs_filepath,
        )


@dataclasses.dataclass(frozen=True)
class EnvVars:
    global_rank: int
    local_rank: int
    world_size: int


@contextlib.contextmanager
def setup(
    seed: int = 42,
    properties: Properties | None = None,
    logs_filepath: pathlib.Path | None = None,
    post_fn: Callable | None = None,
    model: nn.Module | None = None,
    model_has_batchnorm: bool = False,
    dataset: torchvision.datasets.VisionDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    drop_last: bool = False,
    worker_init_fn: Callable | None = None,
) -> Properties:
    start = time.time()

    if properties is None:
        properties = Properties.from_env()

    logger.info(
        (
            "Setting up distributed training on host %s (node %s), "
            "global rank %s, local rank %s (torch.distributed.get_rank()=%s)"
        ),
        socket.gethostname(),
        settings.distributed.node_id,
        settings.distributed.global_rank,
        settings.distributed.local_rank,
        get_rank(settings.distributed),
    )

    utils_logging.create_logger(
        global_rank=properties.global_rank,
        local_rank=properties.local_rank,
        verbose=properties.verbose_logging,
        filepath=properties.logs_filepath,
    )

    if is_primary_device():
        utils_logging.log_env_vars()

        stdout = utils.slurm.get_stdout_file()
        stderr = utils.slurm.get_stderr_file()
        mantik.mlflow.start_run()

    logger.info(
        "Spawning process for node %s, local rank %s, global rank: %s",
        properties.node_id,
        properties.local_rank,
        properties.global_rank,
    )

    _fix_random_seeds(seed, properties=properties)
    _init_process_group(properties)
    _set_device(properties)

    yield properties

    if is_primary_device():
        logger.info("All done!")
        runtime = time.time() - start
        logger.info("Total runtime: %s s", runtime)

        if enable_tracking:
            mantik.mlflow.log_metric("total_runtime_s", runtime)

            _log_stdout_stderr(stdout=stdout, stderr=stderr)
    
    post_fn()

    if is_primary_device():
        mantik.mlflow.end_run()

    destroy()

def _log_stdout_stderr(stdout: str | None, stderr: str | None) -> None:
    if stdout is not None:
        mantik.mlflow.log_artifact(stdout)
    if stderr is not None:
        mantik.mlflow.log_artifact(stderr)

def prepare_model(
    model: nn.Module,
    has_batchnorm: bool,
    properties: Properties,
) -> nn.Module:
    if not properties.use_cpu and has_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    device = get_device(properties)
    # Copy model to GPU
    model = model.to(device)

    if is_primary_device():
        logger.info(model)

    logger.info(
        "Number of trainable parameters: %s",
        utils.models.get_number_of_trainable_parameters(model),
    )
    logger.info(
        "Number of non-trainable parameters: %s",
        utils.models.get_number_of_non_trainable_parameters(model),
    )

    if not properties.use_cpu:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[properties.local_rank],
            find_unused_parameters=True,
        )

    logger.info("Building model done")

    return model, dataloader

def prepare_dataloader(
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int,
    drop_last: bool,
    properties: Properties,
    num_workers: int = 0,
    worker_init_fn: Callable | None = None,
) -> torchvision.datasets.VisionDataset:
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset)
        if properties.use_nccl
        else None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        shuffle=False if properties.world_size > 1 else True,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn or set_dataloader_seeds,
    )
    logger.info(
        "Building data done, dataset size: %s samples", len(dataset)
    )
    logger.info("Batches per epoch: %s", len(dataloader))
    return dataloader

def get_and_set_required_env_vars() -> EnvVars:
    """
    Initialize the following variables:
        - rank
        - local_rank
        - world_size
    """
    logger.info("Visible GPU devices: %i", torch.cuda.device_count())

    if slurm.is_slurm_job():
        logger.info(
            (
                "CUDA_VISIBLE_DEVICES=%s, "
                "SLURM_PROCID=%s, "
                "SLURM_LOCALID=%s, "
                "SLURM_NTASKS=%s"
            ),
            os.environ["CUDA_VISIBLE_DEVICES"],
            os.environ["SLURM_PROCID"],
            os.environ["SLURM_LOCALID"],
            os.environ["SLURM_NTASKS"],
        )

        env_vars = EnvVars(
            global_rank=int(os.environ["SLURM_PROCID"]),
            local_rank=int(os.environ["SLURM_LOCALID"]),
            world_size=int(os.environ["SLURM_NTASKS"]),
        )

        os.environ["RANK"] = str(env_vars.global_rank)
        os.environ["LOCAL_RANK"] = str(env_vars.local_rank)
        os.environ["WORLD_SIZE"] = str(env_vars.world_size)

    else:
        env_vars = EnvVars(
            global_rank=_get_and_set_env_var("RANK", default=0),
            local_rank=_get_and_set_env_var("LOCAL_RANK", default=0),
            world_size=_get_and_set_env_var("WORLD_SIZE", default=1),
        )
    logger.info(
        (
            "Required env vars for distributed mode set: "
            "RANK=%s, LOCAL_RANK=%s, WORLD_SIZE=%s"
        ),
        env_vars.global_rank,
        env_vars.local_rank,
        env_vars.world_size,
    )
    return env_vars


def _get_and_set_env_var(name: str, default: int | str) -> int | str:
    value = os.getenv(name)
    if value is None:
        logger.warning(
            "Environment variable %s unset, using default value %s",
            name,
            default,
        )
        value = default
    else:
        logger.info("Environment variable %s already set to %s", name, value)
        value = type(default)(value)
    os.environ[name] = str(value)
    return value


def _fix_random_seeds(seed: int, properties: Properties) -> None:
    """
    Fix random seeds.
    """
    seed_value = seed + properties.global_rank
    logger.info("Device seed %s", seed_value)

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if not properties.use_cpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_rank(properties: Properties) -> int:
    if properties.use_nccl:
        return torch.distributed.get_rank()
    return 0


def set_dataloader_seeds(_worker_id: int):
    """
    See: https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/  # noqa: E501
    When using "Fork" process spawning, the dataloader workers inherit the
    seeds of the parent process for numpy. While torch seeds are handled
    correctly across dataloaders and across epochs, numpy seeds are not.
    Therefore in order to ensure each worker has a different and deterministic
    seed, we must explicitly set the numpy seed to the torch seed.
    Also see https://pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading  # noqa: E501
    """
    # numpy and random seed must be between 0 and 2 ** 32 - 1.
    torch_seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(torch_seed)
    np.random.seed(torch_seed)


def get_dist_url_and_set_master_env_vars() -> str:
    host = _get_and_set_env_var("MASTER_ADDR", default="127.0.0.1")
    port = _get_and_set_env_var("MASTER_PORT", default=_find_free_tcp_port())

    if not _is_multi_node():
        host = "127.0.0.1"
        logger.info(
            "Assmuning single node environment, setting host to %s:%s",
            host,
            port,
        )
        os.environ["MASTER_ADDR"] = host
    else:
        logger.info(
            "Assuming multi-node environment, host is %s:%s", host, port
        )

    dist_url = f"tcp://{host}:{port}"

    logger.info("Distributed URL is %s", dist_url)
    return dist_url


def _find_free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_multi_gpu() -> bool:
    return int(os.getenv("WORLD_SIZE")) > 1


def _is_multi_node() -> bool:
    return slurm.is_slurm_job() and slurm.get_number_of_nodes() > 1


def _init_process_group(properties: Properties) -> None:
    if not torch.distributed.is_initialized():
        logger.warning(
            "Distributed not initialized, initializing process group"
        )
        if properties.use_cpu:
            logger.info("Initializing CPU backend (gloo)")
            torch.distributed.init_process_group(backend="gloo")
        elif properties.use_nccl:
            logger.warning(
                (
                    "Initializing GPU backend using init_method=%s, "
                    "world_size=%s, rank=%s"
                ),
                properties.dist_url,
                properties.world_size,
                properties.global_rank,
            )
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=properties.dist_url,
                rank=properties.global_rank,
                world_size=properties.world_size,
            )
        else:
            logger.warning(
                "NCCL usage disabled, skipping distributed initialization"
            )
    else:
        logger.warning(
            "Torch distributed has already been initialized, "
            "reusing existing configuration"
        )


def _set_device(properties: Properties) -> None:
    if not properties.use_cpu and torch.cuda.is_available():
        logger.info("Setting torch CUDA device to %s", properties.local_rank)
        torch.cuda.set_device(properties.local_rank)
        # perform a dummy all-reduce to initialize the NCCL communicator
        if properties.use_nccl:
            torch.distributed.all_reduce(torch.zeros(1).cuda())


def get_device(properties: Properties) -> torch.device:
    device = "cpu" if properties.use_cpu else f"cuda:{properties.local_rank}"
    logger.info(
        "Rank %s with local rank %s using device %s",
        properties.global_rank,
        properties.local_rank,
        device,
    )
    return torch.device(device)


def get_single_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info("GPU available, using %s", name)
        # Declare device to be able to copy model and tensors to GPU
        return torch.device("cuda:0")
    logger.info("No GPU available, using CPU instead")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return torch.device("cpu")


def is_primary_device() -> bool:
    return int(os.getenv("RANK")) == 0


def destroy() -> None:
    logger.info("Destroying process group")
    torch.distributed.destroy_process_group()


def gather_from_all_ranks(tensor: torch.Tensor) -> torch.Tensor:
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
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


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

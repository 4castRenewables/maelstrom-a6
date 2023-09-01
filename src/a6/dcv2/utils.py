# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found
# [here](https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/LICENSE)  # noqa: E501
#
import argparse
import logging
import os
import pathlib
import socket
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

import a6.dcv2.logs as logs

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


logger = logging.getLogger(__name__)


def get_device(args):
    return torch.device(
        "cpu" if args.use_cpu else f"cuda:{os.environ['LOCAL_RANK']}"
    )


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with
        # torch.distributed.launch read environment variables
        if "RANK" not in os.environ:
            logger.warning("RANK unset, using default value")
        if "WORLD_SIZE" not in os.environ:
            logger.warning("WORLD_SIZE unset, using default value")
        args.rank = int(os.getenv("RANK", 1))
        args.world_size = int(os.getenv("WORLD_SIZE", 1))

    if args.use_cpu:
        rank = 0
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_find_free_tcp_port())
        dist.init_process_group(backend="gloo")
        args.gpu_to_work_on = rank
    else:
        address = f"{os.environ['SLURMD_NODENAME']}i:29500"
        dist.init_process_group(
            backend="nccl",
            init_method=f"{args.dist_url}{address}",
            world_size=args.world_size,
            rank=args.rank,
        )
        # set cuda device
        args.gpu_to_work_on = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_to_work_on)

    logger.info("Distributed initialized")

    return


def _find_free_tcp_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def initialize_exp(params, *args, dump_params=True):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    if not params.dump_path.exists():
        params.dump_path.mkdir(parents=True, exist_ok=True)

    # dump parameters
    if dump_params:
        with open(params.dump_path / "params.yaml", "w") as f:
            data = {
                key: _path_to_string(value)
                for key, value in vars(params).items()
            }
            yaml.safe_dump(data, stream=f, indent=2, default_flow_style=False)

    # create repo to store checkpoints
    params.dump_checkpoints = params.dump_path / "checkpoints"
    if not params.rank and not params.dump_checkpoints.is_dir():
        params.dump_checkpoints.mkdir(parents=True, exist_ok=True)

    # create a panda object to log loss and acc
    training_stats = logs.Stats(
        params.dump_path / f"stats-rank-{params.rank}.csv", args
    )

    # create a logger
    logger = logs.create_logger(
        params.dump_path / "train.csv", rank=params.rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "%s",
        "\n".join(
            f"{k}: {str(v)}" for k, v in sorted(dict(vars(params)).items())
        ),
    )
    logger.info("The experiment will be stored in %s\n", params.dump_path)
    logger.info("")
    return logger, training_stats


def _path_to_string(value: Any | pathlib.Path) -> Any:
    if isinstance(value, pathlib.Path):
        return value.absolute().as_posix()
    return value


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at %s", ckp_path)

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path,
        map_location="cuda:"
        + str(torch.distributed.get_rank() % torch.cuda.device_count()),
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded %s from checkpoint '%s'", key, ckp_path)
        else:
            logger.warning(
                "=> failed to load %s from checkpoint '%s'", key, ckp_path
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter:
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
    values of k.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

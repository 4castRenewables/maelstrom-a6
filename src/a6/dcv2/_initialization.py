import logging
import pathlib
from typing import Any

import numpy as np
import torch
import yaml

import a6.dcv2.logs as logs

logger = logging.getLogger(__name__)


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


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

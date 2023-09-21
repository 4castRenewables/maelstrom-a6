import logging
import pathlib
from typing import Any

import numpy as np
import torch
import yaml

import a6.dcv2.logs as logs

logger = logging.getLogger(__name__)


def initialize_logging(args, columns) -> tuple[logging.Logger, logs.Stats]:
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """
    # dump parameters
    if not args.dump_path.exists():
        args.dump_path.mkdir(parents=True, exist_ok=True)

    with open(args.dump_path / "args.yaml", "w") as f:
        data = {
            key: _path_to_string(value) for key, value in vars(args).items()
        }
        yaml.safe_dump(data, stream=f, indent=2, default_flow_style=False)

    # create repo to store checkpoints
    args.dump_checkpoints = args.dump_path / "checkpoints"
    if not args.rank and not args.dump_checkpoints.is_dir():
        args.dump_checkpoints.mkdir(parents=True, exist_ok=True)

    args.dump_results = args.dump_path / "results"
    # create repo to store plots
    args.dump_plots = args.dump_results / "plots"
    if not args.rank and not args.dump_plots.is_dir():
        args.dump_plots.mkdir(parents=True, exist_ok=True)

    # create repo to store tensors
    args.dump_tensors = args.dump_results / "tensors"
    if not args.rank and not args.dump_tensors.is_dir():
        args.dump_tensors.mkdir(parents=True, exist_ok=True)

    # create a panda object to log loss and acc
    training_stats = logs.Stats(
        args.dump_path / f"stats-rank-{args.rank}.csv", columns
    )

    # create a logger
    logger_ = logs.create_logger(args.dump_path / "train.csv", args=args)

    if args.rank == 0:
        logger_.info("============ Initialized logging ============")
        logger_.info(
            "%s",
            "\n".join(
                f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())
            ),
        )
        logger_.info("The experiment will be stored in %s\n", args.dump_path)

    return logger, training_stats


def initialize_on_worker(
    args, logger_: logging.Logger, training_stats: logs.Stats
) -> tuple[logging.Logger, logs.Stats]:
    if args.rank == 0:
        return logger_, training_stats
    training_stats = logs.Stats(
        args.dump_path / f"stats-rank-{args.rank}.csv", training_stats.columns
    )
    logger_ = logs.create_logger(args.dump_path / "train.csv", args=args)
    return logger_, training_stats


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

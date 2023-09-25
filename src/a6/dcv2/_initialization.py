import logging
import pathlib
from typing import Any

import yaml

import a6.dcv2.logs as logs

logger = logging.getLogger(__name__)


def initialize_logging(args, columns) -> tuple[logging.Logger, logs.Stats]:
    """Initialize logging.

    Notes
    -----

    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics

    """
    args.dump_checkpoints = args.dump_path / "checkpoints"
    args.dump_results = args.dump_path / "results"
    args.dump_plots = args.dump_results / "plots"
    args.dump_tensors = args.dump_results / "tensors"

    if _is_primary_device(args):
        # dump parameters
        if not args.dump_path.exists():
            args.dump_path.mkdir(parents=True, exist_ok=True)

        with open(args.dump_path / "args.yaml", "w") as f:
            data = {
                key: _path_to_string(value) for key, value in vars(args).items()
            }
            yaml.safe_dump(data, stream=f, indent=2, default_flow_style=False)

        # create repo to store checkpoints
        if not args.global_rank and not args.dump_checkpoints.is_dir():
            args.dump_checkpoints.mkdir(parents=True, exist_ok=True)

        # create repo to store plots
        if not args.global_rank and not args.dump_plots.is_dir():
            args.dump_plots.mkdir(parents=True, exist_ok=True)

        # create repo to store tensors
        if not args.global_rank and not args.dump_tensors.is_dir():
            args.dump_tensors.mkdir(parents=True, exist_ok=True)

    # create a panda object to log loss and acc
    training_stats = logs.Stats(
        args.dump_path / f"stats-rank-{args.global_rank}.csv", columns
    )

    # create a logger
    logger_ = logs.create_logger(
        args.dump_path / f"train-{args.global_rank}.csv", args=args
    )

    if _is_primary_device(args):
        logger_.info("============ Initialized logging ============")
        logger_.info(
            "%s",
            "\n".join(
                f"{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())
            ),
        )
        logger_.info("The experiment will be stored in %s\n", args.dump_path)

    return logger, training_stats


def _is_primary_device(args) -> bool:
    return args.global_rank == 0


def _path_to_string(value: Any | pathlib.Path) -> Any:
    if isinstance(value, pathlib.Path):
        return value.absolute().as_posix()
    return value

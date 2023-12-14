import logging
import os
import pathlib
from typing import Any

import torch.distributed

import a6.utils as utils

logger = logging.getLogger(__name__)


def restart_from_checkpoint(
    paths: pathlib.Path | list[pathlib.Path],
    args: list,
    variables_to_load_from_checkpoint: dict[str, Any] | None = None,
    **kwargs
):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(paths, list):
        for path in paths:
            if os.path.isfile(path):
                break
    else:
        path = paths

    if not os.path.isfile(path):
        return

    logger.info("Found checkpoint at %s", path)

    # open checkpoint file
    checkpoint = torch.load(
        path,
        map_location=utils.distributed.get_device(args),
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
            logger.info("Loaded %s from checkpoint '%s'", key, path)
        else:
            logger.warning("Failed to load %s from checkpoint '%s'", key, path)

    # re load variable important for the run
    if variables_to_load_from_checkpoint is not None:
        for var_name in variables_to_load_from_checkpoint:
            if var_name in checkpoint:
                variables_to_load_from_checkpoint[var_name] = checkpoint[
                    var_name
                ]

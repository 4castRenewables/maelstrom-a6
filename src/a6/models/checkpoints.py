import logging
import pathlib
import shutil
from typing import Any

import torch.distributed
import torch.nn as nn
import torch.optim as optim

import a6.utils as utils

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    path: pathlib.Path,
    checkpoint_freq: int,
    target_epochs: int,
) -> None:
    """Checkpoint the model and optimizer state."""
    file = path / "checkpoint.pth.tar"
    logger.info("Saving checkpoint to %s", file)

    save_dict = {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(
        save_dict,
        file,
    )
    if epoch % checkpoint_freq == 0 or epoch == target_epochs - 1:
        shutil.copyfile(
            file,
            path / f"checkpoint-epoch-{epoch}.pth",
        )


def restart_from_checkpoint(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    properties: utils.distributed.Properties,
    variables_to_load_from_checkpoint: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Re-start from checkpoint.

    `variables_to_load_from_checkpoint` per default contains the last epoch
    as `epoch` key.

    Returns
    -------
    variables_to_load_from_checkpoint : dict, optional
        Contains all given variables that should've been loaded from the
        checkpoint. Per default, this is the last epoch where the model and
        optimizer were checkpointed.

    """

    kwargs = kwargs | {"state_dict": model, "optimizer": optimizer}

    variables_to_load_from_checkpoint = variables_to_load_from_checkpoint or {}
    variables_to_load_from_checkpoint = variables_to_load_from_checkpoint | {
        "epoch": 0
    }

    if not path.is_file():
        return variables_to_load_from_checkpoint

    logger.info("Found checkpoint at %s", path)

    # open checkpoint file
    checkpoint = torch.load(
        path,
        map_location=utils.distributed.get_device(properties),
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

    return variables_to_load_from_checkpoint

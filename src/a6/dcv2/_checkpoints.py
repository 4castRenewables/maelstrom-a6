import logging
import os

import torch.distributed

logger = logging.getLogger(__name__)


def restart_from_checkpoint(ckp_paths, args, run_variables=None, **kwargs):
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

    if not args.use_cpu:
        map_location = (
            f"cuda:{torch.distributed.get_rank() % torch.cuda.device_count()}"
        )
    else:
        map_location = None

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path,
        map_location=map_location,
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

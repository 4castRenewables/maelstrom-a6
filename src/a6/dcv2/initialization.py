import logging

import yaml

import a6.dcv2.settings as _settings
import a6.dcv2.stats as stats

logger = logging.getLogger(__name__)


def initialize(settings: _settings.Settings, columns: list[str]) -> stats.Stats:
    """Initialize dump paths, and dump and log settings.

    Notes
    -----

    - dump parameters
    - create checkpoint repo
    - create a panda object to keep track of the training statistics

    """

    for path in [
        settings.dump.checkpoints,
        settings.dump.results,
        settings.dump.plots,
        settings.dump.tensors,
    ]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    if _is_primary_device(settings):
        # dump parameters
        with open(settings.dump.path / "settings.yaml", "w") as f:
            yaml.safe_dump(
                settings.to_dict(), stream=f, indent=2, default_flow_style=False
            )

    # create a panda object to log loss and acc
    training_stats = stats.Stats(
        settings.dump.path
        / f"stats-rank-{settings.distributed.global_rank}.csv",
        columns,
    )

    if _is_primary_device(settings):
        logger.info("============ Initialized ============")
        logger.info(
            "Settings:\n%s",
            yaml.dump(settings.to_dict(), indent=2, default_flow_style=False),
        )
        logger.info("The experiment will be stored in %s", settings.dump.path)

    return training_stats


def _is_primary_device(settings: _settings.Settings) -> bool:
    return settings.distributed.global_rank == 0

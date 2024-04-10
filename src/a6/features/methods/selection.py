import pandas as pd
import numpy as np
import xarray as xr
import random
import logging

logger = logging.getLogger(__name__)


TrainTestSplit = tuple[list[pd.Timestamp], list[pd.Timestamp]]

def train_test_split_dates(
    date_range: xr.DataArray,
    train_size: int | float,
    continuous: bool = False,
) -> TrainTestSplit:
    """Split given date range into train/test sets."""
    if isinstance(train_size, float):
        train_size = len(date_range) * train_size

    if train_size > len(date_range):
        raise ValueError(
            f"Requested train set size {train_size} is larger than "
            f"date range {date_range}",
        )

    if continuous:
        return _select_continuous(date_range, train_size=train_size)
    return _select_individual(date_range, train_size=train_size)


def _select_continuous(
    date_range: xr.DataArray,
    train_size: int,
) -> TrainTestSplit:
    logger.info(
        (
            "Selecting continuous dates as train/test set "
            "from %s with %i train samples"
        ),
        date_range,
        train_size,
    )

    n_samples = len(date_range)

    start = random.randint(0, n_samples)
    end = start + train_size

    if end > n_samples:
        end = n_samples
        start = n_samples - train_size

    logger.info(
        (
            "Selecting indexes %i to %i as train set, "
            "and 0 to %i and %i to -1 as test set"
        ),
        start,
        end,
        start,
        end,
    )

    train = date_range[start:end]
    test = xr.concat((date_range[:start], date_range[end:]), dim=date_range.dims[0])

    if len(train) == 0:
        raise ValueError(
            "Empty train dataset (start=%s, end=%s)",
            start,
            end
        )
    elif len(test) == 0:
        raise ValueError(
            "Empty test dataset (start=%s, end=%s) from date range %s",
            start,
            end
        )
    return train, test

def _select_individual(
    date_range: xr.DataArray,
    train_size: int,
) -> TrainTestSplit:
    logger.info(
        (
            "Selecting individual dates as train/test set "
            "from %s with %i train samples"
        ),
        date_range,
        train_size,
    )
    shuffled = np.random.permutation(len(date_range))
    train_indexes, test_indexes = sorted(shuffled[:train_size]), sorted(shuffled[train_size:])
    return date_range[train_indexes], date_range[test_indexes]

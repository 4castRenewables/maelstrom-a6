import logging

import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.utils as utils

logger = logging.getLogger(__name__)


@utils.log_consumption
def set_nans_to_mean(
    data: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Mask all values of a given dataset with the mean.

    Assumes the dataset to have a single time step.

    Will set NaN's in a variable at a given level to the mean of that variable
    at the respective level.


    """
    if (n_nans := count_nans(data)) == 0:
        return data

    logger.info("Masking %s NaNs in data set %s to mean", n_nans, data)

    time_steps = data[coordinates.time]

    if time_steps.size > 1:
        for step in time_steps:
            _set_nans_to_mean_for_all_levels_and_variables(
                data.sel({coordinates.time: step}), coordinates=coordinates
            )
    else:
        _set_nans_to_mean_for_all_levels_and_variables(
            data, coordinates=coordinates
        )

    if count_nans(data) != 0:
        raise RuntimeError(
            f"Failed to mask all NaNs with mean for data set {data}"
        )

    return data


def count_nans(data: xr.Dataset) -> int:
    return sum(np.isnan(data[var]).sum() for var in data.data_vars)


def _set_nans_to_mean_for_all_levels_and_variables(
    data: xr.Dataset, coordinates: _coordinates.Coordinates
) -> xr.Dataset:
    levels = data[coordinates.level]

    if levels.size > 1:
        for level in data[coordinates.level]:
            sub = data.sel({coordinates.level: level})
            _set_nans_to_mean_for_all_variables(sub)
    else:
        _set_nans_to_mean_for_all_variables(data)
    return data


def _set_nans_to_mean_for_all_variables(data: xr.Dataset) -> xr.Dataset:
    for var in data.data_vars:
        sub = data[var]
        nans = np.isnan(sub)
        if nans.any():
            sub.values[nans] = sub.mean()
    return data

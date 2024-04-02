import datetime
import logging
from collections.abc import Hashable

import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.features.methods as methods
import a6.types as types
import a6.utils as utils

logger = logging.getLogger(__name__)


@utils.make_functional
def select_variables(
    dataset: xr.Dataset,
    *,
    variables: list[Hashable],
) -> xr.Dataset:
    """Select given variable(s) from the dataset."""
    return dataset[variables]


@utils.make_functional
def select_closest_time_step(
    dataset: xr.Dataset,
    *,
    index: datetime.datetime | np.datetime64 | str,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Select the closest time step."""
    # `method="ffill"` selects closest backwards timestep
    return dataset.sel({coordinates.time: index}, method="ffill")


@utils.make_functional
def select_levels(
    dataset: xr.Dataset,
    *,
    levels: types.Levels,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Select given level(s) from the dataset."""
    return dataset.sel({coordinates.level: levels})


@utils.make_functional
def select_latitude_longitude(
    dataset: xr.Dataset,
    *,
    latitude: int | float,
    longitude: int | float,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Select given latitude and longitude."""
    values = [latitude, longitude]
    if not all(isinstance(val, int) for val in values) and not all(
        isinstance(val, float) for val in values
    ):
        raise ValueError(
            "Given latitude and longitude must be of same type (int or float), "
            f"{type(latitude)} {type(longitude)} given"
        )

    area = {coordinates.latitude: latitude, coordinates.longitude: longitude}
    if isinstance(latitude, int) and isinstance(longitude, int):
        return dataset.isel(area)
    return dataset.sel(area)


@utils.make_functional
def select_levels_and_calculate_daily_mean(
    dataset: xr.Dataset,
    *,
    levels: types.Levels,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Select given level(s) from the dataset and calculate daily means."""
    return (
        select_levels(levels=levels)
        >> methods.averaging.calculate_daily_mean(coordinates=coordinates)
    ).apply_to(dataset)


@utils.make_functional
def select_dwd_area(
    dataset: xr.Dataset,
    *,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Return the dataset, but only the DWD area for GWL.

    Notes
    -----
    See https://www.dwd.de/DE/leistungen/wetterlagenklassifikation/beschreibung.html  # noqa

    """
    area = {
        coordinates.latitude: slice(55.76, 41.47),
        coordinates.longitude: slice(0.0, 19.66),
    }
    return dataset.sel(area)


@utils.log_consumption
@utils.make_functional
def select_intersecting_time_steps(
    left: xr.Dataset,
    right: xr.Dataset,
    return_only_left: bool = True,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset | tuple[xr.Dataset, xr.Dataset]:
    """Select the overlapping time steps of the datasets."""
    intersection = utils.get_time_step_intersection(
        left=left,
        right=right,
        coordinates=coordinates,
    )

    if not intersection:
        logger.warning(
            "No intersecting time steps found for left=%s and right=%s",
            left,
            right,
        )
        raise ValueError("No intersection found")

    select = {coordinates.time: intersection}
    if return_only_left:
        return left.sel(select)
    return left.sel(select), right.sel(select)


def select_for_date(
    data: types.XarrayData,
    *,
    date: datetime.datetime,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> types.XarrayData:
    return data.where(
        (
            (data[f'{coordinates.time}.year'] == date.year) 
            & (data[f'{coordinates.time}.month'] == date.month) 
            & (data[f'{coordinates.time}.day'] == date.day)
        ),
        drop=True,
    )
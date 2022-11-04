from collections.abc import Sequence
from typing import TypeVar

import a6.datasets.coordinates as _coordinates
import a6.features.methods as methods
import a6.utils as utils
import xarray as xr


Levels = TypeVar("Levels", int, Sequence[int])


@utils.make_functional
def select_levels(
    dataset: xr.Dataset,
    levels: Levels,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Select given level(s) from the dataset."""
    return dataset.sel({coordinates.level: levels})


@utils.make_functional
def select_levels_and_calculate_daily_mean(
    dataset: xr.Dataset,
    levels: Levels,
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

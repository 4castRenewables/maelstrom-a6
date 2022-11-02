import typing as t

import a6.features.methods as methods
import a6.utils as utils
import xarray as xr


Levels = t.Union[int, t.Sequence[int]]


def select_levels(dataset: xr.Dataset, levels: Levels) -> xr.Dataset:
    """Select given level(s) from the dataset."""
    return dataset.sel(level=levels)


def select_levels_and_calculate_daily_mean(
    dataset: xr.Dataset,
    levels: Levels,
    time_coordinate: str = "time",
) -> xr.Dataset:
    """Select given level(s) from the dataset and calculate daily means."""
    dataset = select_levels(dataset, levels=levels)
    return methods.averaging.calculate_daily_mean(
        dataset, time_coordinate=time_coordinate
    )


def select_dwd_area(
    dataset: xr.Dataset,
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
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

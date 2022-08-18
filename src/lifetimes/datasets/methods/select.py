import typing as t

import lifetimes.utils
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
    return lifetimes.utils.calculate_daily_mean(
        dataset, time_coordinate=time_coordinate
    )

import typing as t

import lifetimes.utils
import xarray as xr


Level = t.Union[int, t.Sequence[int]]


def select_level(dataset: xr.Dataset, level: Level) -> xr.Dataset:
    """Select given level from the dataset."""
    return dataset.sel(level=level)


def select_level_and_calculate_daily_mean(
    dataset: xr.Dataset,
    level: Level,
    time_coordinate: str = "time",
) -> xr.Dataset:
    """Select given level from the dataset and calculate daily means."""
    dataset = select_level(dataset, level=level)
    return lifetimes.utils.calculate_daily_mean(
        dataset, time_coordinate=time_coordinate
    )

from collections.abc import Callable

import numpy as np
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.utils as utils


@utils.make_functional
def calculate_fraction_of_year(
    ds: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Calculates the temporal fraction w.r.t the year for each time step."""
    fraction_of_year = _apply_to_dates(
        _fraction_of_year, ds[coordinates.time].values
    )
    da = xr.DataArray(
        fraction_of_year,
        coords={coordinates.time: ds[coordinates.time]},
        dims=[coordinates.time],
        name="fraction_of_year",
    )
    ds["fraction_of_year"] = da
    return ds


@utils.make_functional
def calculate_fraction_of_day(
    ds: xr.Dataset,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Calculates the temporal fraction w.r.t the day for each time step."""
    fraction_of_day = _apply_to_dates(
        _fraction_of_day, ds[coordinates.time].values
    )
    da = xr.DataArray(
        fraction_of_day,
        coords={coordinates.time: ds[coordinates.time]},
        dims=[coordinates.time],
        name="fraction_of_day",
    )
    ds["fraction_of_day"] = da
    return ds


def _apply_to_dates(
    func: Callable, dates: np.ndarray[np.datetime64]
) -> np.ndarray:
    return np.array(list(map(func, dates)))


def _fraction_of_year(date: np.datetime64) -> float:
    year = date.astype("datetime64[Y]").astype(int) + 1970
    first = np.datetime64(f"{year}-01-01T00:00")
    last = np.datetime64(f"{year}-12-31T23:59:59")
    fraction = (date - first) / (last - first)
    return float(fraction)


def _fraction_of_day(date: np.datetime64) -> float:
    midnight = date.astype("datetime64[D]").astype("datetime64[s]")
    seconds_since_midnight = (
        (date - midnight).astype("timedelta64[s]").astype(int)
    )
    return float(seconds_since_midnight / 86400)

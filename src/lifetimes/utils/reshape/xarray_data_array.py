import itertools
from typing import Optional

import numpy as np
import xarray as xr


def reshape_spatio_temporal_xarray_data_array(
    data: xr.DataArray,
    time_coordinate: str = "time",
    x_coordinate: Optional[str] = None,
    y_coordinate: Optional[str] = None,
) -> np.ndarray:
    """Reshape a `xr.DataArray` that has one temporal and two spatial dimensions.

    Parameters
    ----------
    data : xr.DataArray
        Input data.
    time_coordinate : str, default="time"
        Name of the time coordinate.
    x_coordinate: str, optional
        Name of the x-coordinate.
        If `None`, the spatial data will be flattened by concatenating the rows.
    y_coordinate: str, optional
        Name of the y-coordinate.
        If `None`, the spatial data will be flattened by concatenating the rows.

    Returns
    -------
    np.ndarray
        Reshaped data with the temporal steps as rows and the spatial points
        as columns (i.e., their respective value).
        If the data has t time steps and consists of a (n, m) grid, the
        reshaped data are of shape (t x nm).

    """
    timesteps = data[time_coordinate]
    timesteps_data = data.sel({time_coordinate: timesteps})
    reshaped = _flatten_spatio_temporal_grid_data(
        timeseries=timesteps_data,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )
    return reshaped


def _flatten_spatio_temporal_grid_data(
    timeseries: xr.DataArray,
    x_coordinate: str,
    y_coordinate: str,
) -> np.ndarray:
    if x_coordinate is None and y_coordinate is None:
        return _flatten_timeseries_with_unlabeled_grid_data(timeseries)
    return _flatten_timeseries_with_labeled_grid_data(
        timeseries=timeseries,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


def _flatten_timeseries_with_unlabeled_grid_data(
    timeseries: xr.DataArray,
) -> np.ndarray:
    flattened = [step.values.flatten() for step in timeseries]
    return np.array(flattened)


def _flatten_timeseries_with_labeled_grid_data(
    timeseries: xr.DataArray, x_coordinate: str, y_coordinate: str
) -> np.ndarray:
    flattened = [
        _flatten_labeled_grid_data(
            data=step,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for step in timeseries
    ]
    return np.array(flattened)


def _flatten_labeled_grid_data(
    data: xr.DataArray, x_coordinate: str, y_coordinate: str
) -> np.ndarray:
    x_coordinates = data[x_coordinate]
    y_coordinates = data[y_coordinate]
    # flatten by concatenating rows
    flattened = [
        data.sel({x_coordinate: x, y_coordinate: y})
        for y, x in itertools.product(y_coordinates, x_coordinates)
    ]
    return np.array(flattened)

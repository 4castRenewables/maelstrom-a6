import functools
import typing as t

import lifetimes.utils.reshape._flatten as _flatten
import numpy as np
import xarray as xr

Data = t.Union[xr.Dataset, xr.DataArray]


def reshape_spatio_temporal_xarray_data(
    data: Data,
    time_coordinate: t.Optional[str] = None,
    x_coordinate: t.Optional[str] = None,
    y_coordinate: t.Optional[str] = None,
) -> np.ndarray:
    """Reshape an `xarray` data object that has one temporal and two spatial
    dimensions.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Input data.
    time_coordinate : str, optional
        Name of the time coordinate.
        Must only be provided if the data is not conform to the CF 1.6
        conventions. This convention states that the coordinates (and
        thus data) are in the order time, latitude, longitude, altitude.
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
    if time_coordinate is not None:
        timesteps = data[time_coordinate]
        data = data.sel({time_coordinate: timesteps})
    return _flatten_spatio_temporal_grid_data(
        data,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


@functools.singledispatch
def _flatten_spatio_temporal_grid_data(
    timeseries: xr.DataArray,
    x_coordinate: t.Optional[str],
    y_coordinate: t.Optional[str],
) -> np.ndarray:
    if x_coordinate is None and y_coordinate is None:
        return _flatten.flatten_timeseries_with_unlabeled_grid_data(timeseries)
    return _flatten.flatten_timeseries_with_labeled_grid_data(
        timeseries=timeseries,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


@_flatten_spatio_temporal_grid_data.register
def _(
    timeseries: xr.Dataset,
    x_coordinate: t.Optional[str],
    y_coordinate: t.Optional[str],
) -> np.ndarray:
    if x_coordinate is None and y_coordinate is None:
        return _flatten_dataset_with_unlabeled_grid_data(timeseries)
    return _flatten_dataset_with_labeled_grid_data(
        data=timeseries,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


def _flatten_dataset_with_unlabeled_grid_data(
    data: xr.Dataset,
) -> np.ndarray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = [
        _flatten.flatten_timeseries_with_unlabeled_grid_data(data[var])
        for var in data.data_vars
    ]
    return np.concatenate(flattened, axis=1)


def _flatten_dataset_with_labeled_grid_data(
    data: xr.Dataset, x_coordinate: str, y_coordinate: str
) -> np.ndarray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = [
        _flatten.flatten_timeseries_with_labeled_grid_data(
            timeseries=data[var],
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for var in data.data_vars
    ]
    return np.concatenate(flattened, axis=1)

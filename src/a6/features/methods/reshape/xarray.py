import functools

import a6.features.methods.reshape._flatten as _flatten
import a6.types as types
import a6.utils as utils
import xarray as xr


@utils.log_consumption
@utils.make_functional
def reshape_spatio_temporal_data(
    data: types.XarrayData,
    time_coordinate: str | None = None,
    x_coordinate: str | None = None,
    y_coordinate: str | None = None,
) -> xr.DataArray:
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
    xr.DataArray
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
    data: xr.DataArray,
    x_coordinate: str | None,
    y_coordinate: str | None,
) -> xr.DataArray:
    if x_coordinate is None and y_coordinate is None:
        return _flatten.flatten_timeseries_with_unlabeled_grid_data(data)
    return _flatten.flatten_timeseries_with_labeled_grid_data(
        data=data,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )


@_flatten_spatio_temporal_grid_data.register
def _(
    data: xr.Dataset,
    x_coordinate: str | None,
    y_coordinate: str | None,
) -> xr.DataArray:
    if x_coordinate is None and y_coordinate is None:
        return _flatten.flatten_dataset_with_unlabeled_grid_data(data)

    return _flatten.flatten_dataset_with_labeled_grid_data(
        data=data,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

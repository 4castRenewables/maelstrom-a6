import itertools

import xarray as xr


def flatten_dataset_with_unlabeled_grid_data(
    data: xr.Dataset,
) -> xr.DataArray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = (
        flatten_timeseries_with_unlabeled_grid_data(data[var])
        for var in data.data_vars
    )
    return xr.concat(flattened, dim="dim_1")


def flatten_dataset_with_labeled_grid_data(
    data: xr.Dataset, x_coordinate: str, y_coordinate: str
) -> xr.DataArray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = (
        flatten_timeseries_with_labeled_grid_data(
            data=data[var],
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for var in data.data_vars
    )
    return xr.concat(flattened, dim="dim_1")


def flatten_timeseries_with_unlabeled_grid_data(
    data: xr.DataArray,
) -> xr.DataArray:
    flattened = [step.data.flatten() for step in data]
    return xr.DataArray(flattened)


def flatten_timeseries_with_labeled_grid_data(
    data: xr.DataArray, x_coordinate: str, y_coordinate: str
) -> xr.DataArray:
    flattened = [
        _flatten_labeled_grid_data(
            data=step,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for step in data
    ]
    return xr.DataArray(flattened)


def _flatten_labeled_grid_data(
    data: xr.DataArray, x_coordinate: str, y_coordinate: str
) -> xr.DataArray:
    x_coordinates = data[x_coordinate]
    y_coordinates = data[y_coordinate]
    # flatten by concatenating rows
    flattened = [
        data.sel({x_coordinate: x, y_coordinate: y})
        for y, x in itertools.product(y_coordinates, x_coordinates)
    ]
    return xr.DataArray(flattened)

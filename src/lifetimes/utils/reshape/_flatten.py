import itertools

import numpy as np
import xarray as xr


def flatten_dataset_with_unlabeled_grid_data(
    data: xr.Dataset,
) -> np.ndarray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = [
        flatten_timeseries_with_unlabeled_grid_data(data[var])
        for var in data.data_vars
    ]
    return np.concatenate(flattened, axis=1)


def flatten_dataset_with_labeled_grid_data(
    data: xr.Dataset, x_coordinate: str, y_coordinate: str
) -> np.ndarray:
    """Flatten each data var individually and then concatenate rows."""
    flattened = [
        flatten_timeseries_with_labeled_grid_data(
            data=data[var],
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for var in data.data_vars
    ]
    return np.concatenate(flattened, axis=1)


def flatten_timeseries_with_unlabeled_grid_data(
    data: xr.DataArray,
) -> np.ndarray:
    flattened = [step.data.flatten() for step in data]
    return np.array(flattened)


def flatten_timeseries_with_labeled_grid_data(
    data: xr.DataArray, x_coordinate: str, y_coordinate: str
) -> np.ndarray:
    flattened = [
        _flatten_labeled_grid_data(
            data=step,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
        )
        for step in data
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

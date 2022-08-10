import itertools

import numpy as np
import xarray as xr


def flatten_timeseries_with_unlabeled_grid_data(
    timeseries: xr.DataArray,
) -> np.ndarray:
    flattened = [step.values.flatten() for step in timeseries]
    return np.array(flattened)


def flatten_timeseries_with_labeled_grid_data(
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

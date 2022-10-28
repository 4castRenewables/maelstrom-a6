import itertools

import numpy as np
import xarray as xr


def apply_kernel_to_grid(
    kernel: np.ndarray, grid: xr.DataArray
) -> xr.DataArray:
    """Apply a given kernel on a grid."""
    kernel_height, kernel_width = kernel.shape
    grid_height, grid_width = grid.shape

    new = grid.copy(deep=True)

    height_range = range(0, grid_height, kernel_height)
    width_range = range(0, grid_width, kernel_width)

    for start_row, start_column in itertools.product(height_range, width_range):
        end_row = start_row + kernel_height
        end_column = start_column + kernel_width
        grid_cutout = grid[start_row:end_row, start_column:end_column].values
        kernel_cutout = kernel[: grid_cutout.shape[0], : grid_cutout.shape[1]]
        kernel_application = (
            grid_cutout * kernel_cutout
        ).sum() / kernel_cutout.sum()
        new[start_row:end_row, start_column:end_column] = kernel_application
    return new

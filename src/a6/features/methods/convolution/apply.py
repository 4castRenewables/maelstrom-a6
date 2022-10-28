import itertools

import numpy as np
import xarray as xr


def apply_kernel_to_grid(
    kernel: np.ndarray, grid: xr.DataArray
) -> xr.DataArray:
    """Apply a kernel on a grid."""
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


def apply_average_kernel(
    data: xr.DataArray,
    longitudinal_size: float,
    latitudinal_size: float,
    longitude: str = "longitude",
    latitude: str = "latitude",
) -> xr.DataArray:
    """Apply an averaging kernel on a grid."""
    lats = data[latitude]
    lons = data[longitude]
    start_lat = lats[0]
    start_lon = lons[0]
    end_lat = lats[-1]
    end_lon = lons[-1]
    lats_step = -latitudinal_size
    lons_step = longitudinal_size
    current_lat = start_lat
    current_lon = start_lon
    new_lats = []
    new_lons = []
    vals = []
    while current_lon < end_lon:
        next_lon = current_lon + lons_step
        while current_lat > end_lat:
            next_lat = current_lat + lats_step
            average = data.sel(
                {
                    latitude: slice(current_lat, next_lat),
                    longitude: slice(current_lon, next_lon),
                }
            ).mean()
            vals.append(average)
            new_lats.append(np.mean([current_lat, next_lat]))
            current_lat = next_lat

        new_lons.append(np.mean([current_lon, next_lon]))
        current_lon = next_lon
    return xr.DataArray(
        np.array(vals).reshape((len(new_lats), len(new_lons))),
        coords={
            latitude: new_lats,
            longitude: new_lons,
        },
        attrs=data.attrs,
    )

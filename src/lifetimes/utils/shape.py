import functools

import xarray as xr


@functools.singledispatch
def get_xarray_data_shape(data: xr.Dataset) -> tuple[int, ...]:
    """Get the shape of an `xarray` data object."""
    dims = tuple(data.sizes.values())
    n_vars = len(data.data_vars)
    return *dims, n_vars


@get_xarray_data_shape.register
def _(data: xr.DataArray) -> tuple[int, ...]:
    return data.shape

import xarray as xr


def np_dot(left: xr.DataArray, right: xr.DataArray) -> xr.DataArray:
    """Return the numpy-like dot product of an `xarray.DataArray`.

    Returns
    -------
    xr.DataArray
        numpy-like dot product result with dimensions of `left`.

    Notes
    -----
    numpy and xarray handle the dot product differently. In xarray, only
    equally named dimensions are used for dot products.

    """
    return xr.DataArray(
        left.data.dot(right.data), dims=left.dims, name=left.name
    )

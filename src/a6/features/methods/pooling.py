import functools

import numpy as np
import skimage.measure
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.types as types
import a6.utils as utils


_POOLING_MODES = {
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
}
_DEFAULT_POOLING_MODE = "mean"


@utils.make_functional
@functools.singledispatch
def apply_pooling(
    data: types.XarrayData,
    size: int,
    mode: str = _DEFAULT_POOLING_MODE,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> types.XarrayData:
    """Apply given method to each step in the timeseries."""
    return NotImplemented


@utils.make_functional
@apply_pooling.register
def _(
    data: xr.Dataset,
    size: int,
    mode: str = _DEFAULT_POOLING_MODE,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.Dataset:
    """Apply given method to each step in the timeseries."""
    result = {
        var: apply_pooling(
            size=size,
            mode=mode,
            coordinates=coordinates,
        ).apply_to(data[var])
        for var in data.data_vars
    }
    return _create_copy_with_data_and_reduced_spatial_coordinates(
        original=data, data=result, size=size, coordinates=coordinates
    )


@utils.make_functional
@apply_pooling.register
def _(
    data: xr.DataArray,
    size: int,
    mode: str = _DEFAULT_POOLING_MODE,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
) -> xr.DataArray:
    """Apply given method to each step in the timeseries."""
    result = [
        _apply_pooling(
            data.sel({coordinates.time: step}),
            size=size,
            mode=mode,
        )
        for step in data[coordinates.time]
    ]
    return _create_copy_with_data_and_reduced_spatial_coordinates(
        original=data, data=result, size=size, coordinates=coordinates
    )


def _create_copy_with_data_and_reduced_spatial_coordinates(
    original: types.XarrayData,
    data: list | dict,
    size: int,
    coordinates: _coordinates.Coordinates,
) -> types.XarrayData:
    coords = {
        coordinates.time: original[coordinates.time],
        coordinates.latitude: _calculate_new_coordinates(
            original[coordinates.latitude], size=size
        ),
        coordinates.longitude: _calculate_new_coordinates(
            original[coordinates.longitude], size=size
        ),
    }
    if isinstance(original, xr.DataArray):
        return xr.DataArray(
            data,
            coords=coords,
            dims=original.dims,
            attrs=original.attrs,
        )
    return xr.Dataset(
        data,
        coords=coords,
        attrs=original.attrs,
    )


def _calculate_new_coordinates(
    coordinates: xr.DataArray, size: int
) -> np.ndarray:
    n_blocks = np.ceil(coordinates.size / size) * size
    start = coordinates[0]
    res = coordinates[1] - start
    end = start + n_blocks * res
    arange = np.linspace(start, end, int(n_blocks), endpoint=False)
    return _apply_pooling(arange, size=size)


def _apply_pooling(
    data: types.Data,
    size: float | tuple[float, float],
    mode: str = _DEFAULT_POOLING_MODE,
) -> types.Data:
    """Apply pooling on an image.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        2D data whose size to reduce.
    size : float or tuple[float, float]
        Size of the pooling block.
    mode : str, default="mean"
        Mode to use for the pooling.
        One of `mean`, `median`, `max` or `min`.

    Returns
    -------
    np.ndarray
        Reduced data.

    Notes
    -----
    If the size of the data is not perfectly divisible by the pooling block
    size, the padding value used (`cval`) is according to `mode`. E.g. for
    `mode="mean"`, `cval=np.mean(data)`.

    """
    try:
        func = _POOLING_MODES[mode]
    except KeyError:
        raise ValueError(f"Pooling mode '{mode} not supported")
    return skimage.measure.block_reduce(
        data,
        block_size=size,
        func=func,
        cval=func(data),
    )

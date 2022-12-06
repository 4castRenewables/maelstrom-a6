import functools

import numpy as np
import scipy.ndimage as ndimage
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.features.methods.convolution._kernels as _kernels
import a6.types as types
import a6.utils as utils

_KERNELS = {
    "mean": _kernels.create_mean_kernel,
    "gaussian": _kernels.create_gaussian_kernel,
}


@utils.make_functional
@functools.singledispatch
def apply_kernel(
    data: types.XarrayData,
    kernel: str | np.ndarray,
    size: int,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    **kwargs,
) -> types.XarrayData:
    """Apply given method to each step in the timeseries."""
    return NotImplemented


@utils.make_functional
@apply_kernel.register
def _(
    data: xr.Dataset,
    kernel: str | np.ndarray,
    size: int,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    **kwargs,
) -> xr.Dataset:
    """Apply given method to each step in the timeseries."""
    result = {
        var: apply_kernel(
            kernel=kernel,
            size=size,
            coordinates=coordinates,
            **kwargs,
        ).apply_to(data[var])
        for var in data.data_vars
    }
    return _create_copy_with_data(original=data, data=result)


@utils.make_functional
@apply_kernel.register
def _(
    data: xr.DataArray,
    kernel: str | np.ndarray,
    size: int,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    **kwargs,
) -> xr.DataArray:
    """Apply given method to each step in the timeseries."""
    result = [
        _apply_kernel(
            data.sel({coordinates.time: step}),
            kernel=kernel,
            size=size,
            **kwargs,
        )
        for step in data[coordinates.time]
    ]
    return _create_copy_with_data(original=data, data=result)


def _create_copy_with_data(
    original: types.XarrayData, data: list | dict
) -> types.XarrayData:
    return original.copy(deep=True, data=data)


def _apply_kernel(
    data: types.Data, kernel: str | np.ndarray, **kwargs
) -> np.ndarray:
    """Apply a given kernel to the data.

    Padding values are filled with the nearest value (`mode="nearest"`).

    """
    if isinstance(kernel, str):
        try:
            factory = _KERNELS[kernel]
        except KeyError:
            raise ValueError(f"Kernel of type '{kernel} not supported")
        kernel = factory(**kwargs)

    return ndimage.convolve(data, kernel, mode="nearest") / kernel.sum()

import functools

import numpy as np
import scipy.ndimage as ndimage
import xarray as xr

import a6.datasets.coordinates as _coordinates
import a6.types as types
import a6.utils as utils


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


def create_mean_kernel(size: int) -> np.ndarray:
    """Create a rectangular, normalized kernel for mean.

    Parameters
    ----------
    size : int
        Width and height of the kernel.

    """
    _check_size_is_odd(size)
    return np.ones((size, size), dtype=np.float32)


def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a gaussian kernel."""
    _check_size_is_odd(size)
    half_length = (size - 1) / 2.0
    x = np.linspace(-half_length, half_length, size)
    gauss_curve = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss_curve, gauss_curve)
    return kernel


def _check_size_is_odd(size: int) -> None:
    if size % 2 == 0:
        raise ValueError("Size of kernel must an odd number")


_KERNELS = {
    "mean": create_mean_kernel,
    "gaussian": create_gaussian_kernel,
}

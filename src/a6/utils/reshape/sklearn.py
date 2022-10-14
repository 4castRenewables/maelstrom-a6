import a6.types as types
import numpy as np
import xarray as xr


def transpose(*data: types.Data) -> np.ndarray:
    """Transpose a given 1D dataset according to the sklearn interface."""
    if len(data) == 1:
        return _reshape(data[0])
    return np.array(list(zip(data)))


def _reshape(data: types.Data) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        return data.values.reshape(-1, 1)
    return data.reshape(-1, 1)

import typing as t

import numpy as np
import xarray as xr

Data = t.Union[np.ndarray, xr.DataArray]
XarrayData = t.Union[xr.DataArray, xr.Dataset]
DataND = t.Union[np.ndarray, xr.DataArray, xr.Dataset]

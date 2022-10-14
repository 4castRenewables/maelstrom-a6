import typing as t

import numpy as np
import xarray as xr

Data = t.Union[np.ndarray, xr.DataArray]
DataND = t.Union[np.ndarray, xr.DataArray, xr.Dataset]

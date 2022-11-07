import a6.utils as utils
import numpy as np
import xarray as xr


@utils.make_functional
def calculate_ssr(left: xr.DataArray, right: xr.DataArray) -> float:
    """Calculate the sum of squared residuals between two datasets."""
    return float(((left - right) ** 2).sum().values)


@utils.make_functional
def calculate_normalized_root_ssr(
    left: xr.DataArray, right: xr.DataArray
) -> float:
    """Calculate the normalized root SSR between two datasets."""
    ssr = calculate_ssr(left, right, non_functional=True)
    return np.sqrt(ssr) / left.max()

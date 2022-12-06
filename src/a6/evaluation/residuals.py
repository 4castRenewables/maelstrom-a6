import numpy as np
import xarray as xr

import a6.utils as utils


@utils.make_functional
def calculate_ssr(y_pred: xr.DataArray, y_true: xr.DataArray) -> float:
    """Calculate the sum of squared residuals between two datasets."""
    return float(((y_pred - y_true) ** 2).sum().values)


@utils.make_functional
def calculate_normalized_root_ssr(
    y_pred: xr.DataArray, y_true: xr.DataArray
) -> float:
    """Calculate the normalized root SSR between two datasets.

    The SSR is normalized to the maximum value of `y_true`.

    """
    ssr = calculate_ssr(y_pred=y_pred, y_true=y_true, non_functional=True)
    return np.sqrt(ssr) / y_true.max()

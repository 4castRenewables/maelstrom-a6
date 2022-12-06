import numpy as np
import sklearn.preprocessing as preprocessing
import xarray as xr

import a6.types as types
import a6.utils as utils


@utils.make_functional
def normalize(data: types.DataND) -> types.DataND:
    """Normalize to zero mean and unit variance (standard deviation).

    Normalizing to zero mean and unit variance is important when using more
    than 1 variable.

    """
    mean_subtracted = data - np.nanmean(data)
    normalized = mean_subtracted / np.nanstd(mean_subtracted)
    return normalized


@utils.make_functional
def normalize_features(data: types.DataND) -> xr.DataArray:
    """Normalize features of a (n_samples x n_features) dataset."""
    scaler = preprocessing.StandardScaler(
        with_mean=True, with_std=True, copy=False
    )
    data: np.ndarray = scaler.fit_transform(data)
    return xr.DataArray(data)

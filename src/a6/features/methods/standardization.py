import a6.types as types
import a6.utils as utils
import numpy as np
import sklearn.preprocessing as preprocessing
import xarray as xr


@utils.make_functional
def standardize(data: types.DataND) -> types.DataND:
    """Standardize to zero mean and unit variance (standard deviation).

    Standardizing to zero mean and unit variance is important when using more
    than 1 variable.

    """
    mean_subtracted = data - np.nanmean(data)
    standardized = mean_subtracted / np.nanstd(mean_subtracted)
    return standardized


@utils.make_functional
def standardize_features(data: types.DataND) -> xr.DataArray:
    """Standardize features of a (n_samples x n_features) dataset."""
    scaler = preprocessing.StandardScaler(
        with_mean=True, with_std=True, copy=False
    )
    data: np.ndarray = scaler.fit_transform(data)
    return xr.DataArray(data)

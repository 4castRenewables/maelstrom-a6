import a6.types as types
import numpy as np
import sklearn.preprocessing as preprocessing
import xarray as xr


def standardize(data: types.Data) -> types.Data:
    """Standardize to zero mean and unit variance (standard deviation).

    Standardizing to zero mean and unit variance is important when using more
    than 1 variable.

    """
    mean_subtracted = data - np.nanmean(data)
    standardized = mean_subtracted / np.nanstd(mean_subtracted)
    return standardized


def standardize_features(data: types.Data) -> xr.DataArray:
    """Standardize features of a (n_samples x n_features) dataset."""
    scaler = preprocessing.StandardScaler(
        with_mean=True, with_std=True, copy=False
    )
    data: np.ndarray = scaler.fit_transform(data)
    return xr.DataArray(data)

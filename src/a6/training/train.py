import sklearn.base
import xarray as xr

import a6.types as types
import a6.utils as utils


def fit(
    model: types.Model,
    X: xr.DataArray,  # noqa: N803
    y: xr.DataArray,
) -> sklearn.base.RegressorMixin:
    """Train a given model."""
    return model.fit(X=utils.transpose(X), y=utils.transpose(y))  # noqa: N803

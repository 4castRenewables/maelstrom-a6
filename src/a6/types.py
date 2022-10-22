import typing as t

import hdbscan
import numpy as np
import sklearn.cluster as cluster
import xarray as xr

Data = t.Union[np.ndarray, xr.DataArray]
XarrayData = t.Union[xr.DataArray, xr.Dataset]
DataND = t.Union[np.ndarray, xr.DataArray, xr.Dataset]
ClusterAlgorithm = t.Union[cluster.KMeans, hdbscan.HDBSCAN]


class MetricEstimator(t.Protocol):
    """Estimates a metric given a ground truth and a prediction."""

    def __call__(self, y_true: Data, y_pred: Data) -> float:
        """Calculate a certain metric."""


class Model(t.Protocol):
    """Estimates a metric given a ground truth and a prediction."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """Fit the model to the given data."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a prediction."""


Scorers = dict[str, MetricEstimator]

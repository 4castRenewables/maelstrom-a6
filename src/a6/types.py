from collections.abc import Sequence
from typing import Protocol
from typing import TypeVar

import hdbscan
import numpy as np
import sklearn.cluster as cluster
import torch
import xarray as xr

Data = TypeVar("Data", np.ndarray, xr.DataArray)
XarrayData = TypeVar("XarrayData", xr.DataArray, xr.Dataset)
DataND = TypeVar("DataND", np.ndarray, xr.DataArray, xr.Dataset, torch.Tensor)
ClusterAlgorithm = TypeVar("ClusterAlgorithm", cluster.KMeans, hdbscan.HDBSCAN)
Levels = TypeVar("Levels", int, Sequence[int], None)
TimeSeries = TypeVar("TimeSeries", xr.DataArray, np.ndarray, Sequence)


class MetricEstimator(Protocol):
    """Estimates a metric given a ground truth and a prediction."""

    def __call__(self, y_true: Data, y_pred: Data) -> float:
        """Calculate a certain metric."""


class Model(Protocol):
    """Estimates a metric given a ground truth and a prediction."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":  # noqa: N803
        """Fit the model to the given data."""

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Make a prediction."""


Scorers = dict[str, MetricEstimator]

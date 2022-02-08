import abc
import typing as t

import numpy as np
import xarray as xr
from sklearn import cluster

from . import pca as _pca


class ClusterAlgorithm(abc.ABC):
    """Wrapper for `sklearn.cluster` algorithms."""

    pca: _pca.PCA

    @property
    def centers(self) -> np.ndarray:
        """Return the cluster centers."""
        return self._centers

    @property
    def labels(self) -> xr.DataArray:
        """Return the labels of each data point."""
        return self._labels

    @property
    @abc.abstractmethod
    def _centers(self) -> np.ndarray:
        ...

    @property
    @abc.abstractmethod
    def _labels(self) -> xr.DataArray:
        ...


class KMeans(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.KMeans`."""

    def __init__(self, kmeans: cluster.KMeans, pca: _pca.PCA, n_components: int):
        """Set attributes.

        Parameters
        ----------
        kmeans : sklearn.cluster.KMeans
            The result of the K-means clustering.
        pca : lifetimes.modes.methods.pca.PCA
            The result of the PCA with the selected number of PCs.
        n_components : int
            Number of PCs.

        """
        self._n_components = n_components
        self._kmeans = kmeans
        self.pca = pca

    @property
    def _centers(self) -> np.ndarray:
        return self._kmeans.cluster_centers_

    @property
    def _labels(self) -> xr.DataArray:
        timeseries = self.pca.timeseries
        return xr.DataArray(
            data=self._kmeans.labels_,
            coords={timeseries.name: timeseries},
        )


def find_principal_component_clusters(
    pca: _pca.PCA,
    n_components: t.Optional[int] = None,
    n_clusters: int = 8,
) -> ClusterAlgorithm:
    """Apply a given clustering algorithm on PCs.

    Parameters
    ----------
    pca : lifetimes.modes.methods.pca.PCA
        Result of the PCA.
    n_components : int, optional
        Number of PCs to use for the clustering.
        Represents the number of dimension of the subspace to perform the
        clustering.
        If `None`, the full PC space will be used.
    n_clusters : int, default=8
        Number of clusters to find.

    Returns
    -------
    KMeans
        Result of the K-means.

    """
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    components_subspace = pca.transform(n_components=n_components)[
        pca._data_variable_name
    ]
    result: cluster.KMeans = kmeans.fit(components_subspace)
    return KMeans(kmeans=result, pca=pca, n_components=n_components)

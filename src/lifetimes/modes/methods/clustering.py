import abc
import typing as t

import hdbscan
import lifetimes.utils
import numpy as np
import xarray as xr
from sklearn import cluster

from . import pca as _pca

_DBSCAN = t.Union[cluster.DBSCAN, hdbscan.HDBSCAN]
_ClusterAlgorithm = t.Union[cluster.KMeans, _DBSCAN]


class ClusterAlgorithm(abc.ABC):
    """Wrapper for `sklearn.cluster`-like algorithms."""

    def __init__(
        self, model: _ClusterAlgorithm, pca: _pca.PCA, n_components: int
    ):
        """Set attributes.

        Parameters
        ----------
        model : KMeans or DBSCAN or HDBSCAN
            The clustering model.
        pca : lifetimes.modes.methods.pca.PCA
            The result of the PCA with the selected number of PCs.
        n_components : int
            Number of PCs.

        """
        self._n_components = n_components
        self._model = model
        self.pca = pca

    @property
    @abc.abstractmethod
    def model(self) -> _ClusterAlgorithm:
        """Return the model."""
        return self._model

    @property
    def centers(self) -> np.ndarray:
        """Return the cluster centers."""
        return self._model.cluster_centers_

    @property
    def labels(self) -> xr.DataArray:
        """Return the labels of the clusters."""
        timeseries = self.pca.timeseries
        return xr.DataArray(
            data=self._model.labels_,
            coords={timeseries.name: timeseries},
        )


class KMeans(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.KMeans`."""

    def __init__(
        self, kmeans: cluster.KMeans, pca: _pca.PCA, n_components: int
    ):
        super().__init__(model=kmeans, pca=pca, n_components=n_components)

    @property
    def model(self) -> cluster.KMeans:
        return super().model


class DBSCAN(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.DBSCAN` or `hdbscan.HDBSCAN`."""

    def __init__(self, dbscan: _DBSCAN, pca: _pca.PCA, n_components: int):
        super().__init__(pca=pca, n_components=n_components)
        self._dbscan = dbscan

    @property
    def model(self) -> _DBSCAN:
        return super().model


@lifetimes.utils.log_runtime
def find_principal_component_clusters(
    pca: _pca.PCA,
    use_varimax: bool = False,
    n_components: t.Optional[int] = None,
    n_clusters: int = 8,
    **clustering_kwargs,
) -> ClusterAlgorithm:
    """Apply a given clustering algorithm on PCs.

    Parameters
    ----------
    pca : lifetimes.modes.methods.pca.PCA
        Result of the PCA.
    use_varimax : bool, default=False
        Whether to perform varimax rotation before the clustering.
    n_components : int, optional
        Number of PCs to use for the clustering.
        Represents the number of dimension of the subspace to perform the
        clustering.
        If `None`, the full PC space will be used.
    n_clusters : int, default=8
        Number of clusters to find.
    kwargs
        Will be passed to the clustering algorithm.

    Returns
    -------
    KMeans
        Result of the K-means.

    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, **clustering_kwargs)
    if use_varimax:
        components_subspace = pca.transform_with_varimax_rotation(
            n_components=n_components
        )
    else:
        components_subspace = pca.transform(n_components=n_components)
    result: cluster.KMeans = kmeans.fit(components_subspace)
    return KMeans(kmeans=result, pca=pca, n_components=n_components)

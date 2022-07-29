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
    def model(self) -> _ClusterAlgorithm:
        """Return the model."""
        return self._model

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

    @property
    def centers(self) -> np.ndarray:
        """Return the cluster centers."""
        return self._model.cluster_centers_


class DBSCAN(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.DBSCAN` or `hdbscan.HDBSCAN`."""


@lifetimes.utils.log_runtime
def find_principal_component_clusters(
    algorithm: _ClusterAlgorithm,
    pca: _pca.PCA,
    use_varimax: bool = False,
    n_components: t.Optional[int] = None,
) -> ClusterAlgorithm:
    """Apply a given clustering algorithm on PCs.

    Parameters
    ----------
    algorithm : KMeans or DBSCAN or HDBSCAN
        The clustering algorithm.
    pca : lifetimes.modes.methods.pca.PCA
        Result of the PCA.
    use_varimax : bool, default=False
        Whether to perform varimax rotation before the clustering.
    n_components : int, optional
        Number of PCs to use for the clustering.
        Represents the number of dimension of the subspace to perform the
        clustering.
        If `None`, the full PC space will be used.

    Returns
    -------
    KMeans
        Result of the clustering algorithm's `fit` method

    """
    if use_varimax:
        components_subspace = pca.transform_with_varimax_rotation(
            n_components=n_components
        )
    else:
        components_subspace = pca.transform(n_components=n_components)
    result: _ClusterAlgorithm = algorithm.fit(components_subspace)
    if isinstance(algorithm, cluster.KMeans):
        return KMeans(model=result, pca=pca, n_components=n_components)
    return DBSCAN(model=result, pca=pca, n_components=n_components)

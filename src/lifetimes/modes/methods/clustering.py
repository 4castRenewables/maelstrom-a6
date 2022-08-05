import abc
from typing import Optional
from typing import Union

import hdbscan.plots
import lifetimes.modes.methods.pca as _pca
import lifetimes.utils
import numpy as np
import xarray as xr
from sklearn import cluster

_ClusterAlgorithm = Union[cluster.KMeans, hdbscan.HDBSCAN]


class ClusterAlgorithm(abc.ABC):
    """Wrapper for `sklearn.cluster`-like algorithms."""

    def __init__(
        self, model: _ClusterAlgorithm, pca: _pca.PCA, n_components: int
    ):
        """Set attributes.

        Parameters
        ----------
        model : KMeans or HDBSCAN
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

    @property
    def n_clusters(self) -> int:
        """Return the number of clusters."""
        # Labelling of clusters starts at index 0
        return self.labels.values.max() + 1


class KMeans(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.KMeans`."""

    @property
    def centers(self) -> np.ndarray:
        """Return the cluster centers."""
        return self._model.cluster_centers_


class HDBSCAN(ClusterAlgorithm):
    """Wrapper for `hdbscan.HDBSCAN`."""

    @property
    def condensed_tree(self) -> hdbscan.plots.CondensedTree:
        """Return the cluster tree."""
        return self.model.condensed_tree_


@lifetimes.utils.log_runtime
def find_principal_component_clusters(
    pca: _pca.PCA,
    use_varimax: bool = False,
    n_components: Optional[int] = None,
    algorithm: Optional[_ClusterAlgorithm] = None,
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
    algorithm : KMeans or HDBSCAN, default=hdbscan.HDBSCAN
        The clustering algorithm.

    Raises
    ------
    NotImplementedError
        If the given clustering algorithm hasn't been implemented yet.

    Returns
    -------
    ClusterAlgorithm
        Result of the clustering algorithm's `fit` method

    """
    if algorithm is None:
        algorithm = hdbscan.HDBSCAN()

    if use_varimax:
        components_subspace = pca.transform_with_varimax_rotation(
            n_components=n_components
        )
    else:
        components_subspace = pca.transform(n_components=n_components)
    result: _ClusterAlgorithm = algorithm.fit(components_subspace)
    if isinstance(algorithm, cluster.KMeans):
        return KMeans(model=result, pca=pca, n_components=n_components)
    elif isinstance(algorithm, hdbscan.HDBSCAN):
        return HDBSCAN(model=result, pca=pca, n_components=n_components)
    return ClusterAlgorithm(model=result, pca=pca, n_components=n_components)

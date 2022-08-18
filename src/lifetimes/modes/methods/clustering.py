import abc
import typing as t

import hdbscan.plots
import lifetimes.modes.methods.pca as _pca
import lifetimes.utils
import numpy as np
import sklearn.cluster as cluster
import xarray as xr

_ClusterAlgorithm = t.Union[cluster.KMeans, hdbscan.HDBSCAN]


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
        self._pca = pca

    @property
    def pca(self) -> _pca.PCA:
        """Return the PCA."""
        return self._pca

    @property
    def n_components(self) -> int:
        """Return the number of PCs used for transformation prior to the
        clustering.
        """
        return self._n_components

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

    @property
    def centers(self) -> xr.DataArray:
        """Return the cluster centers."""
        return xr.DataArray(
            self._centers,
            dims=["cluster", "component"],
        )

    @property
    @abc.abstractmethod
    def _centers(self) -> np.ndarray:
        """Return the cluster centers."""


class KMeans(ClusterAlgorithm):
    """Wrapper for `sklearn.cluster.KMeans`."""

    @property
    def _centers(self) -> np.ndarray:
        return self._model.cluster_centers_


class HDBSCAN(ClusterAlgorithm):
    """Wrapper for `hdbscan.HDBSCAN`."""

    @property
    def _centers(self) -> np.ndarray:
        return np.array(list(self._get_weighted_centers()))

    def _get_weighted_centers(self) -> t.Iterator[list]:
        return (
            self.model.weighted_cluster_centroid(i)
            for i in range(self.n_clusters)
        )

    @property
    def condensed_tree(self) -> hdbscan.plots.CondensedTree:
        """Return the cluster tree."""
        return self.model.condensed_tree_

    def inverse_transformed_cluster(self, cluster_id: int) -> xr.Dataset:
        """Return the inverse transformed cluster.

        The result represents the cluster center in real dimensions of the
        original data, but transformed back with as many PCs as used for the
        clustering. As a consequence, the data may not be identical to the
        original data since it's missing some of the original data's variance.

        """
        center = xr.DataArray(self.model.weighted_cluster_centroid(cluster_id))
        return self.pca.inverse_transform(
            center, n_components=self.n_components
        )


@lifetimes.utils.log_runtime
def find_pc_space_clusters(
    pca: _pca.PCA,
    use_varimax: bool = False,
    n_components: t.Optional[int] = None,
    algorithm: t.Optional[_ClusterAlgorithm] = None,
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

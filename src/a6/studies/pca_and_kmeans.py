import typing as t

import a6.features
import a6.modes
import a6.studies._shared as _shared
import a6.utils as utils
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import xarray as xr

import mlflow


@utils.log_consumption
def perform_pca_and_kmeans(
    data: xr.Dataset,
    n_components: t.Union[int, float],
    n_clusters: int,
    use_varimax: bool,
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
    log_to_mantik: bool = True,
) -> list:
    """Run PCA and K-means on ECMWF IFS HRES data.

    Parameters
    ----------
    data : xr.Dataset
        The data to perform the method on.
    n_components : int or float
        Number of PCs or variance ratio to cover with the PCs for the
        clustering.
    n_clusters : int
        Number of clusters to separate the data into.
    use_varimax : bool
        Whether to use varimax rotation for the PCs before clustering.
    coordinates : a6.utils.CoordinateNames
        Names of the coordinates.
    log_to_mantik : bool, default=True
        Whether to log to mantik.

    Returns
    -------
    list[a6.modes.methods.Mode]
        All modes and their lifetime statistics.

    """
    mlflow_context, log_context = _shared.get_contexts(log_to_mantik)

    with mlflow_context(), log_context():
        pca = a6.modes.methods.spatio_temporal_pca(
            data=data,
            algorithm=decomposition.PCA(n_components=n_components),
            coordinates=coordinates,
        )

        clusters = a6.modes.methods.find_pc_space_clusters(
            pca=pca,
            algorithm=cluster.KMeans(n_clusters=n_clusters),
            use_varimax=use_varimax,
        )

        cluster_a6 = a6.modes.methods.determine_a6_of_modes(
            modes=clusters.labels,
            coordinates=coordinates,
        )

        if log_to_mantik:
            _shared.log_to_mantik(
                pca=pca,
                clusters=clusters,
                n_components=n_components,
                use_varimax=use_varimax,
            )
            mlflow.log_metric("n_kmeans_iterations", clusters.model.n_iter_)
            mlflow.sklearn.log_model(clusters.model, "model")

    return cluster_a6

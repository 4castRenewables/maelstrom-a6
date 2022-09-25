import typing as t

import a6.modes.methods as methods
import a6.modes.methods.clustering as clustering
import a6.modes.methods.pca as _pca
import a6.studies._shared as _shared
import a6.utils as utils
import hdbscan
import numpy as np
import sklearn.decomposition as decomposition
import xarray as xr

import mlflow


@utils.log_consumption
def perform_temporal_range_study(
    data: xr.Dataset,
    n_components: int,
    min_cluster_size: int,
    use_varimax: bool = False,
    log_to_mantik: bool = True,
    coordinates: utils.CoordinateNames = utils.CoordinateNames(),
) -> list[clustering.ClusterAlgorithm]:
    """Perform a temporal range study where the number of time steps used
    increases logarithmically (to the base of 2).
    """
    mlflow_context = _shared.get_mlflow_context(log_to_mantik)
    log_context = _shared.get_logs_context(log_to_mantik)
    clusters = []
    steps = _create_logarithmic_time_slices(
        data, n_components=n_components, coordinates=coordinates
    )

    for step in steps:
        data_subset = _select_data_subset(
            data=data, slice_=step, coordinates=coordinates
        )

        with mlflow_context(), log_context():
            pca = _pca.spatio_temporal_pca(
                data=data_subset,
                algorithm=decomposition.PCA(n_components=n_components),
                coordinates=coordinates,
            )
            clusters_temp = methods.find_pc_space_clusters(
                algorithm=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size),
                pca=pca,
                n_components=n_components,
                use_varimax=use_varimax,
            )
            clusters.append(clusters_temp)

            if log_to_mantik:
                mlflow.log_param("n_time_steps", step.stop)
                _shared.log_to_mantik(
                    pca=pca,
                    clusters=clusters_temp,
                    n_components=n_components,
                    min_cluster_size=min_cluster_size,
                )
    return clusters


def _create_logarithmic_time_slices(
    data: xr.Dataset, n_components: int, coordinates: utils.CoordinateNames
) -> t.Iterator[slice]:
    time_steps = data[coordinates.time].size
    exponent = np.log2(time_steps)
    rounded = int(np.ceil(exponent))
    return (
        slice(None, 2**i) if i < rounded else slice(None, None)
        for i in range(1, rounded + 1)
        if 2**i >= n_components
    )


def _select_data_subset(
    data: xr.Dataset, slice_: slice, coordinates: utils.CoordinateNames
) -> xr.Dataset:
    return data.isel({coordinates.time: slice_})

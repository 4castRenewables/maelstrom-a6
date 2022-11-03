import a6.datasets.coordinates as _coordinates
import a6.modes.methods as methods
import a6.modes.methods.clustering as clustering
import a6.modes.methods.pca as _pca
import a6.studies._shared as _shared
import a6.studies.hyperparameters as _hyperparameters
import a6.types as types
import a6.utils as utils
import sklearn.decomposition as decomposition
import xarray as xr

import mlflow


@utils.log_consumption
def perform_pca_and_cluster_hyperparameter_study(  # noqa: CFQ002
    data: xr.Dataset,
    algorithm: type[types.ClusterAlgorithm],
    hyperparameters: _hyperparameters.HyperParameters,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    use_varimax: bool = False,
    vary_data_variables: bool = False,
    log_to_mantik: bool = True,
) -> list[clustering.ClusterAlgorithm]:
    """Do a hyperparameter study for a PC using different combinations of
    variables of the dataset.
    """
    if not vary_data_variables:
        return _perform_cluster_hyperparameter_study(
            data=data,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            coordinates=coordinates,
            use_varimax=use_varimax,
            log_to_mantik=log_to_mantik,
        )

    clusters = []

    for i, _ in enumerate(data.data_vars):
        data_subset = _select_data_subset(data=data, index=i)
        clusters_temp = _perform_cluster_hyperparameter_study(
            data=data_subset,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            coordinates=coordinates,
            use_varimax=use_varimax,
            log_to_mantik=log_to_mantik,
        )
        clusters.extend(clusters_temp)
    return clusters


@utils.log_consumption
def _perform_cluster_hyperparameter_study(
    data: xr.Dataset,
    algorithm: type[types.ClusterAlgorithm],
    hyperparameters: _hyperparameters.HyperParameters,
    coordinates: _coordinates.Coordinates = _coordinates.Coordinates(),
    use_varimax: bool = False,
    log_to_mantik: bool = True,
) -> list[clustering.ClusterAlgorithm]:
    """Do a hyperparameter study for the HDBSCAN clustering."""
    mlflow_context, log_context = _shared.get_contexts(log_to_mantik)
    clusters = []

    pca = _pca.spatio_temporal_pca(
        data=data,
        algorithm=decomposition.PCA(
            n_components=hyperparameters.n_components_max
        ),
        coordinates=coordinates,
    )

    for (
        n_components,
        cluster_param,
    ) in hyperparameters.to_range():
        with mlflow_context(), log_context():

            clusters_temp = methods.find_pc_space_clusters(
                algorithm=hyperparameters.apply(
                    algorithm,
                    cluster_param,
                ),
                pca=pca,
                n_components=n_components,
                use_varimax=use_varimax,
            )
            clusters.append(clusters_temp)

            if log_to_mantik:
                mlflow.log_param("data_vars", list(data.data_vars))
                _shared.log_to_mantik(
                    pca=pca,
                    clusters=clusters_temp,
                    n_components=n_components,
                    **{hyperparameters.cluster_arg: cluster_param},
                )
    return clusters


def _select_data_subset(data: xr.Dataset, index: int) -> xr.Dataset:
    vars_subset = list(data.data_vars)[: index + 1]
    return data[vars_subset]

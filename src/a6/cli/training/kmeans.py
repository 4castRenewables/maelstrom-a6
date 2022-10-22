import pathlib

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies
import sklearn.cluster as cluster


@train.train.command("kmeans")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.data.VARY_VARIABLES
@_options.pca.N_COMPONENTS_START
@_options.pca.N_COMPONENTS_END
@_options.pca.USE_VARIMAX
@_options.cluster.N_CLUSTERS_START
@_options.cluster.N_CLUSTERS_END
@_options.main.PASS_OPTIONS
def cluster_with_kmeans(
    options: _options.main.Options,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    vary_data_variables: bool,
    n_components_start: int,
    n_components_end: _options.pca.ComponentsEnd,
    use_varimax: bool,
    n_clusters_start: int,
    n_clusters_end: _options.cluster.ClustersEnd,
):
    """Train a KMeans model on PCA results."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        level=level,
    )

    hyperparameters = studies.HyperParameters(
        n_components_start=n_components_start,
        n_components_end=n_components_end,
        cluster_arg="n_clusters",
        cluster_start=n_clusters_start,
        cluster_end=n_clusters_end,
    )

    studies.perform_pca_and_cluster_hyperparameter_study(
        data=ds,
        algorithm=cluster.KMeans,
        vary_data_variables=vary_data_variables,
        hyperparameters=hyperparameters,
        use_varimax=use_varimax,
        log_to_mantik=options.log_to_mantik,
    )

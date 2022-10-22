import pathlib

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies
import hdbscan


@train.train.command("hdbscan")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.data.VARY_VARIABLES
@_options.pca.N_COMPONENTS_START
@_options.pca.N_COMPONENTS_END
@_options.pca.USE_VARIMAX
@_options.cluster.MIN_CLUSTER_SIZE_START
@_options.cluster.MIN_CLUSTER_SIZE_END
@_options.main.PASS_OPTIONS
def cluster_with_hdbscan(
    options: _options.main.Options,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    vary_data_variables: bool,
    n_components_start: int,
    n_components_end: _options.pca.ComponentsEnd,
    min_cluster_size_start: int,
    min_cluster_size_end: _options.cluster.MinClusterSizeEnd,
    use_varimax: bool,
):
    """Train an HDBSCAN model on PCA results."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        level=level,
    )

    hyperparameters = studies.HyperParameters(
        n_components_start=n_components_start,
        n_components_end=n_components_end,
        cluster_arg="min_cluster_size",
        cluster_start=min_cluster_size_start,
        cluster_end=min_cluster_size_end,
    )

    studies.perform_pca_and_cluster_hyperparameter_study(
        data=ds,
        algorithm=hdbscan.HDBSCAN,
        vary_data_variables=vary_data_variables,
        hyperparameters=hyperparameters,
        use_varimax=use_varimax,
        log_to_mantik=options.log_to_mantik,
    )

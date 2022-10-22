import pathlib

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies


@train.train.command("temporal-study")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.pca.N_COMPONENTS
@_options.pca.USE_VARIMAX
@_options.cluster.MIN_CLUSTER_SIZE
@_options.main.PASS_OPTIONS
def perform_temporal_study(
    options: _options.main.Options,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    n_components: _options.pca.Components,
    use_varimax: bool,
    min_cluster_size: int,
):
    """Logarithmically increasing temporal sample size and use HDBSCAN."""
    options.exit_if_dry_run()
    ds = data.read(
        path=weather_data,
        level=level,
    )

    studies.perform_temporal_range_study(
        data=ds,
        n_components=n_components,
        min_cluster_size=min_cluster_size,
        use_varimax=use_varimax,
        log_to_mantik=options.log_to_mantik,
    )

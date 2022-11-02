import pathlib

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies


@train.train.command("cluster")
@_options.data.WEATHER_DATA
@_options.data.PATTERN
@_options.data.SLICE
@_options.data.LEVEL
@_options.data.VARY_VARIABLES
@_options.config.CONFIG
@_options.pca.USE_VARIMAX
@_options.main.PASS_OPTIONS
def train_cluster(  # noqa: CFQ002
    options: _options.main.Options,
    weather_data: pathlib.Path,
    filename_pattern: str,
    slice_weather_data_files: bool,
    level: _options.data.Level,
    vary_data_variables: bool,
    config: _options.config.Config,
    use_varimax: bool,
):
    """Train a cluster model on PCA results."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        pattern=filename_pattern,
        slice_files=slice_weather_data_files,
        level=level,
    )

    hyperparameters = studies.HyperParameters.from_config(config)

    studies.perform_pca_and_cluster_hyperparameter_study(
        data=ds,
        algorithm=config.model,
        vary_data_variables=vary_data_variables,
        hyperparameters=hyperparameters,
        use_varimax=use_varimax,
        log_to_mantik=options.log_to_mantik,
    )

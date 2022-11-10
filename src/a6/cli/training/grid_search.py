import pathlib

import xarray as xr

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies


@train.train.command("grid-search")
@_options.data.WEATHER_DATA
@_options.data.PATTERN
@_options.data.SLICE
@_options.data.LEVEL
@_options.data.TURBINE_DATA
@_options.config.CONFIG
@_options.main.PASS_OPTIONS
def grid_search(  # noqa: CFQ002
    options: _options.main.Options,
    weather_data: pathlib.Path,
    filename_pattern: str,
    slice_weather_data_files: bool,
    level: _options.data.Level,
    turbine_data: pathlib.Path,
    config: _options.config.Config,
):
    """Perform a grid search with the given data."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        pattern=filename_pattern,
        slice_files=slice_weather_data_files,
        level=level,
    )

    turbine = xr.open_dataset(turbine_data)

    studies.perform_forecast_model_grid_search(
        model=config.model,
        parameters=config.parameters,
        weather=ds,
        turbine=turbine,
        log_to_mantik=options.log_to_mantik,
    )

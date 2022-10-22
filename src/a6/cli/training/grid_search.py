import datetime
import pathlib

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies
import xarray as xr


@train.train.command("grid-search")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.data.TURBINE_DATA
@_options.main.PASS_OPTIONS
def grid_search(
    options: _options.main.Options,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    turbine_data: pathlib.Path,
):
    """Perform a grid search with the given data."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        level=level,
        time=slice(
            datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 2, 23)
        ),
    )

    turbine = xr.open_dataset(turbine_data)

    studies.perform_forecast_model_grid_search(
        weather=ds,
        turbine=turbine,
        log_to_mantik=options.log_to_mantik,
    )

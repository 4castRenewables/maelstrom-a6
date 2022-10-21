import datetime
import logging
import pathlib

import a6.cli.data as data
import a6.cli.main as main
import a6.studies as studies
import a6.utils as utils
import click
import xarray as xr


@main.cli.command("grid-search")
@main.WEATHER_DATA
@main.LEVEL
@click.option(
    "--turbine-data",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Local or remote path to the wind turbine data",
)
@main.LOG_TO_MANTIK
@main.GROUP_OPTIONS
def grid_search(
    options: main.Options,
    weather_data: pathlib.Path,
    level: main.Level,
    turbine_data: pathlib.Path,
    log_to_mantik: bool,
):
    """Perform a grid search with the given data."""
    options.exit_if_dry_run()
    utils.log_to_stdout(logging.DEBUG)

    weather = data.read_ecmwf_ifs_hres_data(
        path=weather_data,
        level=level,
    ).sel(
        time=slice(
            datetime.datetime(2017, 1, 1), datetime.datetime(2017, 1, 2, 23)
        )
    )

    turbine = xr.open_dataset(turbine_data)

    studies.perform_forecast_model_grid_search(
        weather=weather,
        turbine=turbine,
        log_to_mantik=log_to_mantik,
    )

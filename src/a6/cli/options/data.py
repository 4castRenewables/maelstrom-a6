import pathlib
import typing as t

import a6.cli.options._callbacks as _callbacks
import click

Level = t.Optional[int]

WEATHER_DATA = click.option(
    "--weather-data",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Local or remote path to the weather data",
)

LEVEL = click.option(
    "-l",
    "--level",
    type=click.UNPROCESSED,
    callback=_callbacks.cast_optional(int),
    required=False,
    default=None,
    help="Level to select from the weather data",
)


TURBINE_DATA = click.option(
    "--turbine-data",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="Local or remote path to the wind turbine data",
)

VARY_VARIABLES = click.option(
    "--vary-data-variables",
    type=click.BOOL,
    default=False,
    required=False,
    show_default=True,
    help="Whether to vary the data variables for a hyperparamater study.",
)

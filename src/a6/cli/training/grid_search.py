import importlib
import pathlib
import typing as t

import a6.cli.data as data
import a6.cli.options as _options
import a6.cli.training.train as train
import a6.studies as studies
import a6.types as types
import click
import xarray as xr
import yaml


@train.train.command("grid-search")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.data.TURBINE_DATA
@_options.main.PASS_OPTIONS
@click.option(
    "-c",
    "--config",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    help="""
        Absolute path to the YAML configuration file for the grid search.

        \b
        Example:
        ```yaml
        model: ensemble.GradientBoostingRegressor
        parameters:
          learning_rate: [0.1, 0.01]
          n_estimators: [10, 50, 100]
        ```
    """,
)
def grid_search(
    options: _options.main.Options,
    weather_data: pathlib.Path,
    level: _options.data.Level,
    turbine_data: pathlib.Path,
    config: pathlib.Path,
):
    """Perform a grid search with the given data."""
    options.exit_if_dry_run()

    ds = data.read(
        path=weather_data,
        level=level,
    )

    turbine = xr.open_dataset(turbine_data)

    config = _read_config(config)

    studies.perform_forecast_model_grid_search(
        model=_get_model_from_config(config),
        parameters=_get_parameters_from_config(config),
        weather=ds,
        turbine=turbine,
        log_to_mantik=options.log_to_mantik,
    )


def _read_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_model_from_config(config: dict) -> t.Type[types.Model]:
    import_statement: str = config["model"]
    module, model = import_statement.rsplit(".", 1)
    sklearn = importlib.import_module(module)
    return getattr(sklearn, model)


def _get_parameters_from_config(config: dict) -> dict:
    return config["parameters"]

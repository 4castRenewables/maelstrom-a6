import dataclasses
import importlib
import pathlib

import click
import yaml

import a6.types as types


@dataclasses.dataclass(frozen=True)
class Config:
    """A config for a hyperparameter study."""

    model: type[types.Model]
    parameters: dict

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "Config":
        """Read from YAML file."""
        config = _read_config(path)
        return cls(
            model=_get_model_from_config(config),
            parameters=_get_parameters_from_config(config),
        )


def _read_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _get_model_from_config(config: dict) -> type[types.Model]:
    import_statement: str = config["model"]
    module, model = import_statement.rsplit(".", 1)
    sklearn = importlib.import_module(module)
    return getattr(sklearn, model)


def _get_parameters_from_config(config: dict) -> dict:
    return config["parameters"]


def parse_config(ctx, param: str, value: pathlib.Path) -> Config:
    return Config.from_yaml(value)


CONFIG = click.option(
    "-c",
    "--config",
    type=click.Path(path_type=pathlib.Path),
    callback=parse_config,
    required=True,
    help="""
        Absolute path to the YAML configuration file for the model and input
        parameters.

        \b
        Example:
        ```yaml
        model: sklearn.ensemble.GradientBoostingRegressor
        parameters:
          learning_rate: [0.1, 0.01]
          n_estimators: [10, 50, 100]
        ```
    """,
)

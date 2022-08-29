import argparse
import ast
import os
import typing as t

import mantik.utils.mlflow as mlflow


def create_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for execution via the mantik Compute Backend.

    Parameters
    ----------
    parser : argparse.ArgumentParser, optional
        Parser to which the arguments will be added.

    Returns
    -------
    parser : argparse.ArgumentParser
        Arguments can be retrieved via `parser.parse_args()`.
        Additional arguments can be added via `parser.add_argument()`.
        Default arguments that will be available as attributes after parsing:
            entry_point : str
                Entry point of the MLproject to execute.
            mlflow_parameters : list[tuple(str, Any)]
                Parameters to pass to the entry point.
            experiment_id : int
                Experiment ID of the experiment to track to.

    """
    if parser is None:
        parser = argparse.ArgumentParser(
            "Execute an MLproject entry point on HPC with the Compute Backend."
        )

    parser.add_argument(
        "-e",
        "--entry-point",
        type=str,
        required=True,
        help="The entry point of the MLproject to execute.",
    )

    parser.add_argument(
        "-P",
        "--param",
        type=_parse_key_value_pair,
        required=True,
        dest="mlflow_parameters",
        action="append",
        metavar="KEY=VALUE",
        help="Parameters to pass to the MLflow entry point.",
    )
    parser.add_argument(
        "--experiment-id",
        type=int,
        default=None,
        required=False,
        help="ID of the MLflow experiment to track to.",
    )
    parser.add_argument(
        "--backend-config",
        type=str,
        default="unicore-config.json",
        required=False,
        help="Name of the config for the Compute Backend.",
    )
    return parser


def _parse_key_value_pair(kv: t.Any) -> t.Tuple[str, t.Any]:
    key, value = kv.split("=", 1)
    return key, _parse_value(value)


def _parse_value(value: t.Any) -> t.Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If value is a string, `astr.literal_eval` raises ValueError
        # and in some cases a SyntaxError.
        try:
            return ast.literal_eval(f"'{value}'")
        except ValueError:
            raise argparse.ArgumentTypeError(f"Unable to parse {value}")


def read_experiment_id_from_env() -> int:
    """Read the experiment ID from an environment variable."""
    experiment_id = os.getenv(mlflow.EXPERIMENT_ID_ENV_VAR)
    if experiment_id is None:
        raise NameError(
            f"Environment variable '{mlflow.EXPERIMENT_ID_ENV_VAR}' "
            "must be set or the `--experiment-id` argument given."
        )
    return int(experiment_id)

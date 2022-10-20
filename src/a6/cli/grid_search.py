import argparse
import typing as t

import a6.cli.conversion as conversion


def create_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for grid search.

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
            data : str
                Path to the data.
            level : int, optional
                Level to choose from the data.
            log_to_mantik : bool, default=True
                Whether to log to mantik.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Perform a grid search.")

    parser.add_argument(
        "--weather-data",
        type=str,
        required=True,
        help="Local or remote path to the weather data.",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=conversion.cast_optional(int),
        default=None,
        required=False,
        help="Level to choose from the data.",
    )
    parser.add_argument(
        "--turbine-data",
        type=str,
        required=True,
        help="Local or remote path to the wind turbine data.",
    )
    parser.add_argument(
        "--log-to-mantik",
        type=conversion.string_to_bool,
        default=True,
        required=False,
        help="Whether to log to mantik.",
    )
    return parser

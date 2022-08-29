import argparse
import typing as t

import lifetimes.cli.conversion as conversion


def create_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for inference with a model.

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
            variance_ratios : float
                Variance ratio to use for PCA.
            use_varimax : bool
                Whether to use varimax rotation for the PCs.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Use a model for inference.")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Local or remote path to the data.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        required=True,
        help="Variance ratio to use for PCA.",
    )
    parser.add_argument(
        "--use-varimax",
        type=conversion.string_to_bool,
        default=False,
        required=False,
        help="Whether to use varimax rotation for the PCs.",
    )
    return parser

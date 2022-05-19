import argparse
import typing as t

import lifetimes.cli.conversion as conversion


def create_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for model training.

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
            variance_ratios : list[float]
                Variance ratios to use for PCA.
            n_clusters : list[int]
                Number of clusters to use for the clustering algorithm.
            use_varimax : list[bool]
                Whether to use varimax rotation for the PCs.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Train a model.")
    parser.add_argument(
        "--data", type=str, help="Local or remote path to the data."
    )
    parser.add_argument(
        "--variance-ratios",
        nargs="+",
        type=float,
        help="Variance ratio to use for PCA.",
    )
    parser.add_argument(
        "--n-clusters",
        nargs="+",
        type=int,
        help="Number of clusters to use for the clustering algorithm.",
    )
    parser.add_argument(
        "--use-varimax",
        nargs="+",
        type=conversion.string_to_bool,
        help="Whether to use varimax rotation for the PCs.",
    )
    return parser

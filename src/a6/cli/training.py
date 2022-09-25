import argparse
import typing as t

import a6.cli.conversion as conversion


def create_temporal_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for KMeans model training.

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
            n_components : int, default=3
                Number of principal components to use for PCA.
            min_cluster_size : int, default=2
                `min_cluster_size` to use for HDBSCAN.
            use_varimax : list[bool], default=[False]
                Whether to use varimax rotation for the PCs.
            log_to_mantik : bool, default=True
                Whether to log to mantik.
            env_file : str, optional
                Path to the `.env` file with the mantik credentials.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Do a temporal study of HDBSCAN.")
        parser = create_parser(parser)

    parser.add_argument(
        "--n-components",
        type=int,
        default=3,
        required=True,
        help="Number of components to use for PCA.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        required=True,
        help="`min_cluster_size` for the HDBSCAN algorithm.",
    )
    parser.add_argument(
        "--use-varimax",
        type=conversion.string_to_bool,
        default=False,
        required=False,
        help="Whether to use varimax rotation for the PCs.",
    )
    return parser


def create_hdbscan_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for HDBSCAN model training.

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
            n_components_start : int, default=3
                Minimum number of principal components to use for PCA.
            n_components_end : int, optional
                Maximum number of principal components to use for PCA.
                Defaults to `None`, which only performs a single run for
                `n_components=n_components_start`.
            min_cluster_size_start : int, default=None
                Minimum `min_cluster_size` for HDBSCAN.
            min_cluster_size_end : int, optional
                Maximum `min_cluster_size` for HDBSCAN.
                Defaults to `None`, which only performs a single run for
                `min_cluser_size=min_cluster_size_start`.
            use_varimax : list[bool], default=[False]
                Whether to use varimax rotation for the PCs.
            log_to_mantik : bool, default=True
                Whether to log to mantik.
            env_file : str, optional
                Path to the `.env` file with the mantik credentials.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Train an HDBSCAN model.")
        parser = create_parser(parser)

    parser.add_argument(
        "--vary-data-variables",
        type=conversion.string_to_bool,
        default=False,
        required=False,
        help="Whether to vary the data variables for the hyperparamater study.",
    )
    parser.add_argument(
        "--n-components-start",
        type=int,
        default=3,
        required=True,
        help="Minimum number of components to use for PCA.",
    )
    parser.add_argument(
        "--n-components-end",
        type=conversion.cast_optional(int),
        default=None,
        required=False,
        help="Maximum number of components to use for PCA.",
    )
    parser.add_argument(
        "--min-cluster-size-start",
        type=int,
        help="Minimum `min_cluster_size` for the HDBSCAN.",
        default=2,
        required=True,
    )
    parser.add_argument(
        "--min-cluster-size-end",
        type=conversion.cast_optional(int),
        default=None,
        required=False,
        help="Maximum `min_cluster_size` for the HDBSCAN.",
    )
    parser.add_argument(
        "--use-varimax",
        type=conversion.string_to_bool,
        default=False,
        required=False,
        help="Whether to use varimax rotation for the PCs.",
    )
    return parser


def create_kmeans_parser(
    parser: t.Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    """Create CLI parser for KMeans model training.

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
            n_components : list[int], default=[3]
                Number of principal components to use for PCA.
            n_clusters : list[int], default=[4]
                Number of clusters to use for the clustering algorithm.
            use_varimax : list[bool], default=[False]
                Whether to use varimax rotation for the PCs.
            log_to_mantik : bool, default=True
                Whether to log to mantik.
            env_file : str, optional
                Path to the `.env` file with the mantik credentials.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Train a KMeans model.")
        parser = create_parser(parser)

    parser.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=[3],
        required=True,
        help="Number of components to use for PCA.",
    )
    parser.add_argument(
        "--n-clusters",
        nargs="+",
        type=int,
        default=[4],
        required=False,
        help="Number of clusters to use for the clustering algorithm.",
    )
    parser.add_argument(
        "--use-varimax",
        nargs="+",
        type=conversion.string_to_bool,
        default=[False],
        required=False,
        help="Whether to use varimax rotation for the PCs.",
    )
    return parser


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
            level : int, optional
                Level to choose from the data.
            log_to_mantik : bool, default=True
                Whether to log to mantik.
            env_file : str, optional
                Path to the `.env` file with the mantik credentials.

    """
    if parser is None:
        parser = argparse.ArgumentParser("Train a KMeans model.")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Local or remote path to the data.",
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
        "--log-to-mantik",
        type=conversion.string_to_bool,
        default=True,
        required=False,
        help="Whether to log to mantik.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        required=False,
        help=(
            "Local path to the .env file that contains the environment"
            "variables (credentials) required for mantik to allow tracking."
        ),
    )
    return parser

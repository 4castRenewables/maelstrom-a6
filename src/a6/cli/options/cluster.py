import typing as t

import click

ClustersStart = t.Optional[int]
ClustersEnd = t.Optional[int]
MinClusterSizeEnd = t.Optional[int]

N_CLUSTERS_START = click.option(
    "--n-clusters-start",
    type=int,
    default=2,
    required=False,
    show_default=True,
    help="Minimum `n_clusters` for the KMeans.",
)

N_CLUSTERS_END = click.option(
    "--n-clusters-end",
    type=int,
    default=None,
    required=False,
    show_default=True,
    help="Maximum `n_clusters` for the KMeans.",
)

MIN_CLUSTER_SIZE = click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    required=False,
    show_default=True,
    help="`min_cluster_size` for the HDBSCAN algorithm.",
)

MIN_CLUSTER_SIZE_START = click.option(
    "--min-cluster-size-start",
    type=int,
    default=2,
    required=False,
    show_default=True,
    help="Minimum `min_cluster_size` for the HDBSCAN.",
)

MIN_CLUSTER_SIZE_END = click.option(
    "--min-cluster-size-end",
    type=int,
    default=None,
    required=False,
    show_default=True,
    help="Maximum `min_cluster_size` for the HDBSCAN.",
)

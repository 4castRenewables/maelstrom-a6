import a6.cli.main as main
import a6.cli.options as _options
import click
import pytest


@main.cli.command("test-kmeans-command")
@_options.cluster.N_CLUSTERS_START
@_options.cluster.N_CLUSTERS_END
def kmeans_command(
    n_clusters_start: _options.cluster.ClustersStart,
    n_clusters_end: _options.cluster.ClustersEnd,
):
    click.echo(n_clusters_start)
    click.echo(f"{n_clusters_end}")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--n-clusters-start", "2"], ["2", "None"]),
        (["--n-clusters-start", "2", "--n-clusters-end", "4"], ["2", "4"]),
    ],
)
def test_kmeans_cluster_options(runner, args, expected):
    args = ["test-kmeans-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)


@main.cli.command("test-hdbscan-command")
@_options.cluster.MIN_CLUSTER_SIZE
@_options.cluster.MIN_CLUSTER_SIZE_START
@_options.cluster.MIN_CLUSTER_SIZE_END
def hdbscan_command(
    min_cluster_size: int,
    min_cluster_size_start: int,
    min_cluster_size_end: _options.cluster.MinClusterSizeEnd,
):
    click.echo(min_cluster_size)
    click.echo(min_cluster_size_start)
    click.echo(f"{min_cluster_size_end}")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--min-cluster-size", "2", "--min-cluster-size-start", "3"],
            ["2", "3", "None"],
        ),
        (
            [
                "--min-cluster-size",
                "2",
                "--min-cluster-size-start",
                "3",
                "--min-cluster-size-end",
                "4",
            ],
            ["2", "3", "4"],
        ),
    ],
)
def test_hdbscan_cluster_options(runner, args, expected):
    args = ["test-hdbscan-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)

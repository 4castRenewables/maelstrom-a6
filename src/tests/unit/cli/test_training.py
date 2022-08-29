import lifetimes.cli.training as training
import pytest


@pytest.mark.parametrize(
    (
        "cli_args",
        "expected_data",
        "expected_level",
        "expected_n_components",
        "expected_min_cluster_size",
        "expected_use_varimax",
    ),
    [
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components",
                "3",
                "--min-cluster-size",
                "2",
            ],
            "test-path",
            1,
            3,
            2,
            False,
        ),
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components",
                "3",
                "--min-cluster-size",
                "2",
                "--use-varimax",
                "True",
            ],
            "test-path",
            1,
            3,
            2,
            True,
        ),
    ],
)
def test_temporal_parser(
    cli_args,
    expected_data,
    expected_level,
    expected_n_components,
    expected_min_cluster_size,
    expected_use_varimax,
):
    parser = training.create_temporal_parser()
    args = parser.parse_args(cli_args)

    assert args.data == expected_data
    assert args.level == expected_level
    assert args.n_components == expected_n_components
    assert args.min_cluster_size == expected_min_cluster_size
    assert args.use_varimax == expected_use_varimax


@pytest.mark.parametrize(
    (
        "cli_args",
        "expected_data",
        "expected_vary_data_variables",
        "expected_level",
        "expected_n_components_start",
        "expected_n_components_end",
        "expected_min_cluster_size_start",
        "expected_min_cluster_size_end",
        "expected_use_varimax",
    ),
    [
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components-start",
                "3",
                "--min-cluster-size-start",
                "2",
            ],
            "test-path",
            False,
            1,
            3,
            None,
            2,
            None,
            False,
        ),
        # Test case: Set vary_data_variables `True`
        (
            [
                "--data",
                "test-path",
                "--vary-data-variables",
                "True",
                "--level",
                "1",
                "--n-components-start",
                "3",
                "--min-cluster-size-start",
                "2",
            ],
            "test-path",
            True,
            1,
            3,
            None,
            2,
            None,
            False,
        ),
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components-start",
                "3",
                "--n-components-end",
                "None",
                "--min-cluster-size-start",
                "2",
                "--min-cluster-size-end",
                "None",
            ],
            "test-path",
            False,
            1,
            3,
            None,
            2,
            None,
            False,
        ),
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components-start",
                "3",
                "--n-components-end",
                "4",
                "--min-cluster-size-start",
                "2",
                "--min-cluster-size-end",
                "3",
                "--use-varimax",
                "True",
            ],
            "test-path",
            False,
            1,
            3,
            4,
            2,
            3,
            True,
        ),
    ],
)
def test_hdbscan_parser(
    cli_args,
    expected_data,
    expected_vary_data_variables,
    expected_level,
    expected_n_components_start,
    expected_n_components_end,
    expected_min_cluster_size_start,
    expected_min_cluster_size_end,
    expected_use_varimax,
):
    parser = training.create_hdbscan_parser()
    args = parser.parse_args(cli_args)

    assert args.data == expected_data
    assert args.vary_data_variables == expected_vary_data_variables
    assert args.level == expected_level
    assert args.n_components_start == expected_n_components_start
    assert args.n_components_end == expected_n_components_end
    assert args.min_cluster_size_start == expected_min_cluster_size_start
    assert args.min_cluster_size_end == expected_min_cluster_size_end
    assert args.use_varimax == expected_use_varimax


@pytest.mark.parametrize(
    (
        "cli_args",
        "expected_data",
        "expected_level",
        "expected_n_components",
        "expected_n_clusters",
        "expected_use_varimax",
    ),
    [
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components",
                "3",
                "4",
                "--n-clusters",
                "1",
                "2",
            ],
            "test-path",
            1,
            [3, 4],
            [1, 2],
            False,
        ),
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components",
                "1",
                "--n-clusters",
                "1",
                "--use-varimax",
                "True",
            ],
            "test-path",
            1,
            [1],
            [1],
            [True],
        ),
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
                "--n-components",
                "1",
                "--n-clusters",
                "1",
                "--use-varimax",
                "True",
                "False",
            ],
            "test-path",
            1,
            [1],
            [1],
            [True, False],
        ),
    ],
)
def test_kmeans_parser(
    cli_args,
    expected_data,
    expected_level,
    expected_n_components,
    expected_n_clusters,
    expected_use_varimax,
):
    parser = training.create_kmeans_parser()
    args = parser.parse_args(cli_args)

    assert args.data == expected_data
    assert args.level == expected_level
    assert args.n_components == expected_n_components
    assert args.n_clusters == expected_n_clusters
    assert args.use_varimax == expected_use_varimax


@pytest.mark.parametrize(
    (
        "cli_args",
        "expected_data",
        "expected_level",
        "expected_log_to_mantik",
        "expected_env_file",
    ),
    [
        (
            [
                "--data",
                "test-path",
                "--level",
                "1",
            ],
            "test-path",
            1,
            True,
            None,
        ),
    ],
)
def test_parser(
    cli_args,
    expected_data,
    expected_level,
    expected_log_to_mantik,
    expected_env_file,
):
    parser = training.create_parser()
    args = parser.parse_args(cli_args)

    assert args.data == expected_data
    assert args.level == expected_level
    assert args.log_to_mantik == expected_log_to_mantik
    assert args.env_file == expected_env_file

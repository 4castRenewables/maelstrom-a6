import a6.cli.main as main


def test_cluster_with_hdbscan(runner):
    args = [
        "--dry-run",
        "train",
        "hdbscan",
        "--weather-data",
        "/test/path/",
    ]
    result = runner.invoke(
        main.cli,
        args,
    )

    assert result.exit_code == 0

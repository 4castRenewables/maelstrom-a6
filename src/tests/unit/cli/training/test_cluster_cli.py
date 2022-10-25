import a6.cli.main as main


def test_train_cluster(runner, config_path):
    args = [
        "--dry-run",
        "train",
        "cluster",
        "--weather-data",
        "/test/path/",
        "--config",
        config_path.as_posix(),
    ]
    result = runner.invoke(
        main.cli,
        args,
    )

    assert result.exit_code == 0

import a6.cli.main as main


def test_grid_search(runner, config_path):
    args = [
        "--dry-run",
        "train",
        "grid-search",
        "--weather-data",
        "/test/path/",
        "--turbine-data",
        "/test/other/path/",
        "--config",
        config_path.as_posix(),
    ]
    result = runner.invoke(
        main.cli,
        args,
    )

    assert result.exit_code == 0

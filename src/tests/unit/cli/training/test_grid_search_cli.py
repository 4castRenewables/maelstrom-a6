import a6.cli.main as main


def test_grid_search(runner):
    args = [
        "--dry-run",
        "train",
        "grid-search",
        "--weather-data",
        "/test/path/",
        "--turbine-data",
        "/test/other/path/",
    ]
    result = runner.invoke(
        main.cli,
        args,
    )

    assert result.exit_code == 0

import a6.cli.main as main


def test_temporal(runner):
    args = [
        "--dry-run",
        "train",
        "temporal-study",
        "--weather-data",
        "/test/path/",
        "--n-components",
        "3",
    ]
    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0

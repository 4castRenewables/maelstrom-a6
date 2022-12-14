import a6.cli.main as main


def test_perform_inference(monkeypatch, runner):
    args = [
        "--dry-run",
        "inference",
        "test-aws-endpoint",
        "--weather-data",
        "test-path",
        "--n-components",
        "1",
        "--use-varimax",
        "true",
    ]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0

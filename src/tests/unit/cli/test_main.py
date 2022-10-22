import a6.cli.main as main


def test_dry_run(runner):
    args = ["--dry-run", "--help"]
    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0


def test_log_to_mantik(runner):
    args = ["--log-to-mantik", "--help"]
    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0

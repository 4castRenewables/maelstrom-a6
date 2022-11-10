import click
import pytest

import a6.cli.main as main
import a6.cli.options as _options


@main.cli.command("test-main-command")
@_options.main.PASS_OPTIONS
def main_command(
    options: _options.main.Options,
):
    options.exit_if_dry_run()
    click.echo(options.log_to_mantik)
    click.echo("Success")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--dry-run"], "Dry running"),
        ([], "Success"),
    ],
)
def test_dry_run(runner, args, expected):
    args = [*args, "test-main-command"]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert expected in result.output


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--log-to-mantik", "true"], "True"),
        ([], "False"),
    ],
)
def test_log_to_mantik(runner, args, expected):
    args = [*args, "test-main-command"]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert "Success" in result.output
    assert expected in result.output

import pathlib

import a6.cli.main as main
import click
import pytest


@main.cli.command("test-command")
@main.WEATHER_DATA
@main.LEVEL
@main.LOG_TO_MANTIK
@main.GROUP_OPTIONS
def command(
    options: main.Options,
    weather_data: pathlib.Path,
    level: main.Level,
    log_to_mantik: bool,
):
    options.exit_if_dry_run()
    click.echo("Success")


@pytest.mark.parametrize(
    ("dry_run", "args", "expected_exit_code", "expected_output"),
    [
        (
            False,
            [],
            2,
            None,
        ),
        (True, ["--weather-data", "/some/path"], 0, "Dry running"),
        (False, ["--weather-data", "/some/path"], 0, "Success"),
        (False, ["--weather-data", "/some/path", "-l", "500"], 0, "Success"),
        (
            False,
            ["--weather-data", "/some/path", "--level", "500"],
            0,
            "Success",
        ),
        (
            False,
            ["--weather-data", "/some/path", "--log-to-mantik"],
            0,
            "Success",
        ),
    ],
)
def test_command(runner, dry_run, args, expected_exit_code, expected_output):
    args = ["test-command", *args]

    if dry_run:
        args = ["--dry-run", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == expected_exit_code

    if expected_output is not None:
        assert expected_output in result.output

    if dry_run:
        assert "Success" not in result.output

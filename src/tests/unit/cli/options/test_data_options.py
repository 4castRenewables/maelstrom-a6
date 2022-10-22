import pathlib

import a6.cli.main as main
import a6.cli.options as _options
import click
import pytest


@main.cli.command("test-data-command")
@_options.data.WEATHER_DATA
@_options.data.LEVEL
@_options.data.TURBINE_DATA
@_options.data.VARY_VARIABLES
def data_command(
    weather_data: pathlib.Path,
    level: _options.data.Level,
    turbine_data: pathlib.Path,
    vary_data_variables: bool,
):
    click.echo(f"{level}")
    click.echo(f"{vary_data_variables}")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
            ],
            ["None"],
        ),
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
                "--vary-data-variables",
                "True",
            ],
            ["None", "True"],
        ),
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
                "-l",
                "500",
            ],
            ["500"],
        ),
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
                "--level",
                "500",
            ],
            ["500"],
        ),
    ],
)
def test_data_options(runner, args, expected):
    args = ["test-data-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)

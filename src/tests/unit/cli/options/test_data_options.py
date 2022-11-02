import pathlib

import a6.cli.main as main
import a6.cli.options as _options
import click
import pytest


@main.cli.command("test-data-command")
@_options.data.WEATHER_DATA
@_options.data.PATTERN
@_options.data.SLICE
@_options.data.LEVEL
@_options.data.TURBINE_DATA
@_options.data.VARY_VARIABLES
def data_command(
    weather_data: pathlib.Path,
    filename_pattern: str,
    slice_weather_data_files: bool,
    level: _options.data.Level,
    turbine_data: pathlib.Path,
    vary_data_variables: bool,
):
    click.echo(f"{filename_pattern}")
    click.echo(f"{slice_weather_data_files}")
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
            ["*.nc", "False", "None"],
        ),
        # Test case: slice-weather-data-files option
        (
            [
                "--weather-data",
                "/test/path",
                "-s",
                "true",
                "--turbine-data",
                "/test/other/path",
            ],
            ["True"],
        ),
        # Test case: slice-weather-data-files option
        (
            [
                "--weather-data",
                "/test/path",
                "--slice-weather-data-files",
                "true",
                "--turbine-data",
                "/test/other/path",
            ],
            ["True"],
        ),
        # Test case: pattern option
        (
            [
                "--weather-data",
                "/test/path",
                "-p",
                "test*pattern",
                "--turbine-data",
                "/test/other/path",
            ],
            ["test*pattern"],
        ),
        # Test case: pattern option
        (
            [
                "--weather-data",
                "/test/path",
                "--filename-pattern",
                "test*pattern",
                "--turbine-data",
                "/test/other/path",
            ],
            ["test*pattern"],
        ),
        # Test case: vary-data-variables option
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
                "--vary-data-variables",
                "True",
            ],
            ["True"],
        ),
        # Test case: level option
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
        # Test case: level option
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
        # Test case: level option
        (
            [
                "--weather-data",
                "/test/path",
                "--turbine-data",
                "/test/other/path",
                "--level",
                "None",
            ],
            ["None"],
        ),
    ],
)
def test_data_options(runner, args, expected):
    args = ["test-data-command", *args]

    result = runner.invoke(main.cli, args)

    assert result.exit_code == 0
    assert all(output in result.output for output in expected)

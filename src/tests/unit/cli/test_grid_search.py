import pathlib

import a6.cli.grid_search as grid_search
import pytest


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            [
                "--weather-data",
                "/some/path/",
                "--turbine-data",
                "/some/other/path/",
            ],
            grid_search.GridSearchArguments(
                weather=grid_search._shared.Weather(
                    path=pathlib.Path("/some/path"),
                ),
                turbine=grid_search.Turbine(
                    path=pathlib.Path("/some/other/path"),
                ),
                log_to_mantik=False,
            ),
        ),
    ],
)
def test_arguments(invoke_and_get_return_value, args, expected):
    result = invoke_and_get_return_value(
        grid_search.arguments,
        args,
    )

    assert result == expected

import datetime

import pytest

import a6.plotting.modes.comparison as comparison


@pytest.fixture
def data(pl_ds):
    return pl_ds.sel(level=500)


@pytest.fixture
def dates() -> list[datetime.datetime]:
    return [
        datetime.datetime(2020, 12, 1),
        datetime.datetime(2020, 12, 2),
        datetime.datetime(2020, 12, 3),
    ]


def test_plot_fields_for_dates(data, dates):
    comparison.plot_fields_for_dates(
        field=data["t"],
        dates=dates,
    )


@pytest.mark.parametrize("temperature", [None, "t"])
def test_plot_contours_for_field_and_dates(data, dates, temperature):
    if temperature is not None:
        temperature = data.isel(time=0)["t"]

    comparison.plot_contours_for_field_and_dates(
        field=data["z"],
        dates=dates,
    )


def test_plot_wind_speed_for_dates(data, dates):
    comparison.plot_wind_speed_for_dates(
        field=data,
        dates=dates,
    )


def test_plot_combined(data, dates):
    comparison.plot_combined(
        data=data,
        dates=dates,
        geopotential_height="z",
    )

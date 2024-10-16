import datetime

import numpy as np
import xarray as xr

import a6.datasets.methods.turbine as turbine

COORDS = {
    "latitude": 1.123,
    "longitude": 1.123,
}


def test_clean_production_data():
    power_rating = 1000
    production = _create_dataset(
        values=[100, 1000, 1200, 500, -1, 0],
        dates=[
            datetime.datetime(2022, 1, 1, 0),
            datetime.datetime(2022, 1, 1, 1),
            datetime.datetime(2022, 1, 1, 2),
            datetime.datetime(2022, 1, 1, 3),
            datetime.datetime(2022, 1, 1, 4),
            datetime.datetime(2022, 1, 1, 5),
        ],
    )

    expected = _create_dataset(
        values=[100, 1000, 500],
        dates=[
            datetime.datetime(2022, 1, 1, 0),
            datetime.datetime(2022, 1, 1, 1),
            datetime.datetime(2022, 1, 1, 3),
        ],
    )

    result = turbine.clean_production_data(
        production, power_rating=power_rating, non_functional=True
    )

    xr.testing.assert_equal(result, expected)


def test_resample_to_hourly_resolution():
    production = _create_dataset(
        values=[100, np.NaN, 1000, 500, 1, 1, 1, 1, 1, 1],
        dates=[
            datetime.datetime(2022, 1, 1, 0, 10),
            datetime.datetime(2022, 1, 1, 0, 20),
            datetime.datetime(2022, 1, 1, 1, 0),
            datetime.datetime(2022, 1, 1, 1, 10),
            datetime.datetime(2022, 1, 1, 2, 10),
            datetime.datetime(2022, 1, 1, 2, 20),
            datetime.datetime(2022, 1, 1, 2, 30),
            datetime.datetime(2022, 1, 1, 2, 40),
            datetime.datetime(2022, 1, 1, 2, 50),
            datetime.datetime(2022, 1, 1, 3, 0),
        ],
    )

    expected = _create_dataset(
        values=[((100 + 1000) / 2) * 6, 500 * 6, 1 * 6],
        dates=[
            datetime.datetime(2022, 1, 1, 0),
            datetime.datetime(2022, 1, 1, 1),
            datetime.datetime(2022, 1, 1, 2),
        ],
    )

    result = turbine.resample_to_hourly_resolution(
        production, non_functional=True
    )

    xr.testing.assert_equal(result, expected)


def test_get_closest_grid_point():
    production = _create_dataset(
        values=[100],
        dates=[
            datetime.datetime(2022, 1, 1),
        ],
    )

    weather = xr.DataArray(
        [
            [
                [1, 2],
                [3, 4],
            ]
        ],
        coords={
            "time": [datetime.datetime(2022, 1, 1)],
            "latitude": [1.2, 1.1],
            "longitude": [1.1, 1.2],
        },
    )

    expected = xr.DataArray(
        [3],
        coords={
            "time": [datetime.datetime(2022, 1, 1)],
            "latitude": 1.1,
            "longitude": 1.1,
        },
        dims=["time"],
    )

    result = turbine.get_closest_grid_point(
        weather=weather, turbine=production, non_functional=True
    )

    xr.testing.assert_equal(result, expected)


def _create_dataset(
    values: list,
    dates: list[datetime.datetime],
) -> xr.Dataset:
    production = xr.DataArray(
        values,
        coords={"time": dates},
        name="production",
    )
    return xr.Dataset(
        data_vars={production.name: production},
        coords={
            **production.coords,
            **COORDS,
        },
    )

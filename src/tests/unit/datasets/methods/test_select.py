import datetime

import numpy as np
import pytest
import xarray as xr

import a6.datasets.methods.select as select

COORDS = {
    "latitude": 1.123,
    "longitude": 1.123,
}


@pytest.mark.parametrize("levels", [500, [500, 1000]])
def test_select_levels(pl_ds, levels):
    result = select.select_levels(pl_ds, levels=levels, non_functional=True)

    assert result["level"].values.tolist() == levels


def test_select_levels_and_calculate_daily_mean(pl_ds):
    expected = np.array(
        [
            [
                [245.4356, 245.53772],
                [245.34428, 245.38219],
                [245.47972, 245.4964],
            ],
            [
                [235.93938, 235.96883],
                [235.544, 235.57054],
                [235.26147, 235.30586],
            ],
            [
                [233.545, 233.50867],
                [232.33002, 232.33957],
                [232.06425, 232.16368],
            ],
        ],
        dtype=np.float32,
    )

    result = select.select_levels_and_calculate_daily_mean(
        pl_ds, levels=500, non_functional=True
    )

    assert result["level"].values == 500

    np.testing.assert_allclose(result["t"].values, expected)


def test_select_closest_time_step():
    expected = "2000-01-01"
    da = xr.DataArray(
        [1, 2], dims=["time"], coords={"time": ["2000-01-01", "2000-01-02"]}
    )
    ds = xr.Dataset({"var1": da}, coords=da.coords)
    result = select.select_closest_time_step(
        ds, index="2000-01-01T12:00", non_functional=True
    )

    assert str(result["time"].values) == expected


def test_select_intersecting_time_steps():
    production = _create_dataset(
        values=[2, 3, 4],
        dates=[
            datetime.datetime(2022, 1, 2),
            datetime.datetime(2022, 1, 3),
            datetime.datetime(2022, 1, 4),
        ],
    )
    weather = _create_dataset(
        values=[4, 5, 6],
        dates=[
            datetime.datetime(2022, 1, 1),
            datetime.datetime(2022, 1, 2),
            datetime.datetime(2022, 1, 3),
        ],
    )

    expected_production = _create_dataset(
        values=[2, 3],
        dates=[
            datetime.datetime(2022, 1, 2),
            datetime.datetime(2022, 1, 3),
        ],
    )

    expected_weather = _create_dataset(
        values=[5, 6],
        dates=[
            datetime.datetime(2022, 1, 2),
            datetime.datetime(2022, 1, 3),
        ],
    )

    result_weather, result_production = select.select_intersecting_time_steps(
        left=weather,
        right=production,
        non_functional=True,
    )

    xr.testing.assert_equal(result_weather, expected_weather)
    xr.testing.assert_equal(result_production, expected_production)


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

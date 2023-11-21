import datetime

import xarray as xr

import a6.features.methods.time as time


def test_calculate_fraction_of_year():
    coords = {
        "time": [
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 3, 31),
            datetime.datetime(2000, 6, 30),
            datetime.datetime(2000, 9, 30),
            datetime.datetime(2000, 12, 31),
        ]
    }

    ds = xr.Dataset(
        data_vars={
            "var1": (["time"], [i for i in range(len(coords["time"]))]),
        },
        coords=coords,
    )

    expected = xr.DataArray(
        data=[0.0, 0.245902, 0.494536, 0.745902, 0.997268],
        dims=["time"],
        coords=coords,
        name="fraction_of_year",
    )

    result_ds = time.calculate_fraction_of_year(ds)
    result = result_ds["fraction_of_year"]

    xr.testing.assert_allclose(result, expected)


def test_calculate_fraction_of_day():
    coords = {
        "time": [
            datetime.datetime(2000, 1, 1, 0),
            datetime.datetime(2000, 1, 1, 6),
            datetime.datetime(2000, 1, 1, 12),
            datetime.datetime(2000, 1, 1, 18),
            datetime.datetime(2000, 1, 2, 0),
        ]
    }

    ds = xr.Dataset(
        data_vars={
            "var1": (["time"], [i for i in range(len(coords["time"]))]),
        },
        coords=coords,
    )

    expected = xr.DataArray(
        data=[0.0, 0.25, 0.5, 0.75, 0.0],
        dims=["time"],
        coords=coords,
        name="fraction_of_day",
    )

    result_ds = time.calculate_fraction_of_day(ds)
    result = result_ds["fraction_of_day"]

    xr.testing.assert_allclose(result, expected)

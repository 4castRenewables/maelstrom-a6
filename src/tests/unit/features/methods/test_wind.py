import a6.features.methods.wind as wind
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def coords():
    return {
        "lat": [0, 0.1],
        "lon": [0, 0.1],
    }


@pytest.fixture
def data(coords):
    u = xr.DataArray(
        [
            [1, 2],
            [3, 4],
        ],
        coords=coords,
        name="u",
    )
    v = xr.DataArray(
        [
            [1, 2],
            [3, 4],
        ],
        coords=coords,
        name="v",
    )
    return xr.Dataset(
        data_vars={"u": u, "v": v},
        coords=coords,
    )


def test_calculate_wind_speed(coords, data):
    expected = xr.DataArray(
        [
            [np.sqrt(2), np.sqrt(8)],
            [np.sqrt(18), np.sqrt(32)],
        ],
        coords=coords,
    )

    result = wind.calculate_wind_speed(data)

    xr.testing.assert_equal(result, expected)


def test_calculate_wind_direction_angle(coords, data):
    expected = xr.DataArray(
        [
            [45.0, 45.0],
            [45.0, 45.0],
        ],
        coords=coords,
    )

    result = wind.calculate_wind_direction_angle(data)

    xr.testing.assert_equal(result, expected)

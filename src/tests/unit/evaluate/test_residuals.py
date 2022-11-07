import a6.evaluation.residuals as residuals
import numpy as np
import pandas as pd
import pytest
import xarray as xr

TIME = pd.date_range("2000-01-01", "2000-01-02", freq="1d")
LATS = [1.0, 0.0]
LONS = [0.0, 1.0]

COORDS = {
    "time": TIME,
    "latitude": LATS,
    "longitude": LONS,
}

DIMS = ["time", "latitude", "longitude"]


@pytest.fixture()
def left() -> xr.DataArray:
    return xr.DataArray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [5.0, 6.0],
                [7.0, 8.0],
            ],
        ],
        coords=COORDS,
        dims=DIMS,
    )


@pytest.fixture()
def right() -> xr.DataArray:
    return xr.DataArray(
        [
            [
                [2.0, 4.0],
                [6.0, 8.0],
            ],
            [
                [10.0, 12.0],
                [14.0, 16.0],
            ],
        ],
        coords=COORDS,
        dims=DIMS,
    )


@pytest.fixture()
def expected_ssr() -> float:
    return sum(
        [
            (1.0 - 2.0) ** 2,
            (2.0 - 4.0) ** 2,
            (3.0 - 6.0) ** 2,
            (4.0 - 8.0) ** 2,
            (5.0 - 10.0) ** 2,
            (6.0 - 12.0) ** 2,
            (7.0 - 14.0) ** 2,
            (8.0 - 16.0) ** 2,
        ]
    )


def test_calculate_ssr(left, right, expected_ssr):
    result = residuals.calculate_ssr(left, right, non_functional=True)

    assert result == expected_ssr


def test_calculate_normalized_root_ssr(left, right, expected_ssr):
    expected = np.sqrt(expected_ssr) / 8.0

    result = residuals.calculate_normalized_root_ssr(
        left, right, non_functional=True
    )

    assert result == expected

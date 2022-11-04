import a6.features.methods.pooling as pooling
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def coords() -> dict:
    return {
        "time": pd.date_range("2000-01-01", "2000-01-02", freq="1d"),
        "latitude": [3.0, 2.0, 1.0, 0.0],
        "longitude": [0.0, 1.0, 2.0, 3.0],
    }


@pytest.fixture()
def dims() -> list[str]:
    return ["time", "latitude", "longitude"]


@pytest.fixture()
def data_array(coords, dims) -> xr.DataArray:
    return xr.DataArray(
        [
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            [
                [2, 2, 3, 3],
                [2, 2, 3, 3],
                [4, 4, 5, 5],
                [4, 4, 5, 5],
            ],
        ],
        coords=coords,
        dims=dims,
    )


@pytest.fixture()
def expected_data_array(coords, dims) -> xr.DataArray:
    return xr.DataArray(
        [
            [
                [1, 2],
                [3, 4],
            ],
            [
                [2, 3],
                [4, 5],
            ],
        ],
        coords={
            "time": coords["time"],
            "latitude": [2.5, 0.5],
            "longitude": [0.5, 2.5],
        },
        dims=dims,
    )


def test_apply_pooling_data_array(data_array, expected_data_array):
    result = pooling.apply_pooling(
        data_array, mode="mean", size=2, non_functional=True
    )

    xr.testing.assert_equal(result, expected_data_array)


def test_apply_pooling_dataset(data_array, expected_data_array):
    data = xr.Dataset(
        data_vars={
            "var1": data_array,
            "var2": data_array,
        },
        coords=data_array.coords,
    )
    expected = xr.Dataset(
        data_vars={
            "var1": expected_data_array,
            "var2": expected_data_array,
        },
        coords=expected_data_array.coords,
    )
    result = pooling.apply_pooling(
        data, mode="mean", size=2, non_functional=True
    )

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("coordinates", "size", "expected"),
    [
        (xr.DataArray([3.0, 2.0, 1.0, 0.0]), 2, np.array([2.5, 0.5])),
        (xr.DataArray([0.0, 1.0, 2.0, 3.0]), 2, np.array([0.5, 2.5])),
        (xr.DataArray([4.0, 3.0, 2.0, 1.0]), 2, np.array([3.5, 1.5])),
        (xr.DataArray([1.0, 2.0, 3.0, 4.0]), 2, np.array([1.5, 3.5])),
        (xr.DataArray([0.0, 1.0, 2.0, 3.0, 4.0]), 3, np.array([1.0, 4.0])),
        (xr.DataArray([4.0, 3.0, 2.0, 1.0, 0.0]), 3, np.array([3.0, 0.0])),
        (xr.DataArray([0.0, 1.0, 2.0, 3.0, 4.0]), 4, np.array([1.5, 5.5])),
        (xr.DataArray([4.0, 3.0, 2.0, 1.0, 0.0]), 4, np.array([2.5, -1.5])),
    ],
)
def test_calculate_new_coordinates(coordinates, size, expected):
    result = pooling._calculate_new_coordinates(
        coordinates=coordinates,
        size=size,
    )

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "size", "mode", "expected"),
    [
        (
            np.array([0.0, 1.0, 2.0, 3.0]),
            2,
            "mean",
            np.array([0.5, 2.5]),
        ),
        (
            np.array(
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ]
            ),
            2,
            "mean",
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [0, 1, 0, 2],
                    [1, 2, 2, 3],
                    [0, 3, 0, 4],
                    [3, 4, 4, 5],
                ]
            ),
            2,
            "median",
            np.array(
                [
                    [1, 2],
                    [3, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                ]
            ),
            2,
            "max",
            np.array(
                [
                    [4, 4],
                    [4, 4],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                    [1, 2, 1, 2],
                    [3, 4, 3, 4],
                ]
            ),
            2,
            "min",
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1, 1, 2],
                    [1, 1, 2],
                    [3, 3, 4],
                ]
            ),
            2,
            "mean",
            np.array(
                [
                    [4.0 / 4.0, 8.0 / 4.0],
                    [10.0 / 4.0, 10.0 / 4.0],
                ]
            ),
        ),
    ],
)
def test_apply_pooling(data, size, mode, expected):
    result = pooling._apply_pooling(data, size=size, mode=mode)

    np.testing.assert_equal(result, expected)

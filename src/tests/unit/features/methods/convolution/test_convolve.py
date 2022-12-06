import numpy as np
import pandas as pd
import pytest
import xarray as xr

import a6.features.methods.convolution.convolve as convolve


@pytest.fixture()
def coords() -> dict:
    return {
        "time": pd.date_range("2000-01-01", "2000-01-02", freq="1d"),
        "latitude": [1.0, 0.0],
        "longitude": [0.0, 1.0],
    }


@pytest.fixture()
def dims() -> list[str]:
    return ["time", "latitude", "longitude"]


@pytest.fixture()
def data_array(coords, dims) -> xr.DataArray:
    return xr.DataArray(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            [
                [2.0, 3.0],
                [4.0, 5.0],
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
                [12.0 / 9.0, 12.0 / 9.0],
                [15.0 / 9.0, 15.0 / 9.0],
            ],
            [
                [27.0 / 9.0, 30.0 / 9.0],
                [33.0 / 9.0, 36.0 / 9.0],
            ],
        ],
        coords=coords,
        dims=dims,
    )


def test_apply_kernel_data_array(data_array, expected_data_array):
    result = convolve.apply_kernel(
        data_array, kernel="mean", size=3, non_functional=True
    )

    xr.testing.assert_equal(result, expected_data_array)


def test_apply_kernel_dataset(data_array, expected_data_array):
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
    result = convolve.apply_kernel(
        data, kernel="mean", size=3, non_functional=True
    )

    xr.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "kernel", "kwargs", "expected"),
    [
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            {},
            [
                [12.0 / 9.0, 12.0 / 9.0],
                [15.0 / 9.0, 15.0 / 9.0],
            ],
        ),
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            "mean",
            {"size": 3},
            [
                [12.0 / 9.0, 12.0 / 9.0],
                [15.0 / 9.0, 15.0 / 9.0],
            ],
        ),
        (
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
            [
                [1.0],
                [1.0],
                [1.0],
            ],
            {},
            [
                [4.0 / 3.0, 4.0 / 3.0],
                [6.0 / 3.0, 6.0 / 3.0],
                [9.0 / 3.0, 9.0 / 3.0],
                [11.0 / 3.0, 11.0 / 3.0],
            ],
        ),
    ],
)
def test_apply_kernel(data, kernel, kwargs, expected):
    if isinstance(kernel, list):
        kernel = np.array(kernel)
    result = convolve._apply_kernel(np.array(data), kernel=kernel, **kwargs)

    np.testing.assert_equal(result, np.array(expected))

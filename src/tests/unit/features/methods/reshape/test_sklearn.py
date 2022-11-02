import a6.features.methods.reshape.sklearn as sklearn
import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([np.array([1, 2, 3])], np.array([[1], [2], [3]])),
        (
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            np.array([[1, 4], [2, 5], [3, 6]]),
        ),
        ([xr.DataArray([1, 2, 3])], np.array([[1], [2], [3]])),
        (
            [xr.DataArray([1, 2, 3]), xr.DataArray([4, 5, 6])],
            np.array([[1, 4], [2, 5], [3, 6]]),
        ),
    ],
)
def test_transpose(data, expected):
    result = sklearn.transpose(*data)

    if isinstance(data, np.ndarray):
        np.testing.assert_equal(result, expected)
    elif isinstance(data, xr.DataArray):
        np.testing.assert_equal(result, expected.values)

import numpy as np
import pytest
import xarray as xr
from a6.utils.reshape import xarray_data


@pytest.mark.parametrize(
    ("time_coordinate", "x_coordinate", "y_coordinate"),
    [
        ("time", None, None),
        ("time", "lon", "lat"),
    ],
)
def test_reshape_spatio_temporal_xarray_data_array(
    da, time_coordinate, x_coordinate, y_coordinate
):
    expected = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
    )

    result = xarray_data.reshape_spatio_temporal_xarray_data(
        data=da,
        time_coordinate=time_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    ("time_coordinate", "x_coordinate", "y_coordinate"),
    [
        ("time", None, None),
        ("time", "lon", "lat"),
    ],
)
def test_reshape_spatio_temporal_xarray_dataset(
    da, time_coordinate, x_coordinate, y_coordinate
):

    ds = xr.Dataset(
        data_vars={"var1": da, "var2": da},
        coords=da.coords,
    )

    expected = np.array(
        [
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12],
        ],
    )

    result = xarray_data.reshape_spatio_temporal_xarray_data(
        data=ds,
        time_coordinate=time_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    np.testing.assert_equal(result, expected)

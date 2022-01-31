import datetime

import numpy as np
import pytest
import xarray as xr
from lifetimes.utils.reshape import xarray_data_array


@pytest.mark.parametrize(
    ("time_coordinate", "x_coordinate", "y_coordinate"),
    [
        ("time", None, None),
        ("time", "lon", "lat"),
    ],
)
def test_reshape_spatio_temporal_xarray_data_array(
    time_coordinate, x_coordinate, y_coordinate
):
    data = [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
    ]
    da = xr.DataArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": (
                "time",
                [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)],
            ),
            "lat": ("lat", [0.0, 1.0]),
            "lon": ("lon", [0.0, 1.0, 2.0]),
        },
    )

    expected = np.array(
        [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
    )

    result = xarray_data_array.reshape_spatio_temporal_xarray_data_array(
        data=da,
        time_coordinate=time_coordinate,
        x_coordinate=x_coordinate,
        y_coordinate=y_coordinate,
    )

    np.testing.assert_equal(result, expected)

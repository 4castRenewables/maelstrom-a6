import datetime

import pytest
import xarray as xr


@pytest.fixture()
def da() -> xr.DataArray:
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
    return xr.DataArray(
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

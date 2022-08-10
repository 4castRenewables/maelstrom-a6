import datetime

import lifetimes.testing as testing
import numpy as np
import xarray as xr


def test_generate_ecmwf_ifs_hres_data():
    grid = testing.TestGrid(rows=3, columns=5)
    a = 2 / 3
    b = 2 / 3
    ellipse_1 = testing.EllipticalDataFactory(
        a=a,
        b=b,
    )
    data_points = [
        testing.DataPoints(
            data_factory=ellipse_1,
            start="2000-01-01",
            end="2000-01-01",
            frequency="1d",
        )
    ]
    dataset = testing.FakeEcmwfIfsHresDataset(
        grid=grid,
        start="2000-01-01",
        end="2000-01-02",
        frequency="1d",
        data=data_points,
    )
    expected_data = np.array(
        [
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=float,
    )

    dates = [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)]
    expected = xr.Dataset(
        data_vars={
            "ellipse": (["time", "lat", "lon"], expected_data),
        },
        coords={"time": (["time"], dates), **grid.xarray_coords_dict},
    )
    result = dataset.as_xarray()

    xr.testing.assert_equal(result, expected)
